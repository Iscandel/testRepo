#include <helper_gl.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


//Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>

//Project includes
#include "core/pathtracerKernel.h"
#include "tools/cudaUtils.h"
#include "core/myMath.h"
#include "core/geometry.h"
#include "camera/trackballHelper.h"

#include "core/scene.h"
#include "tools/timer.h"
#include "tools/tools.h"

#define BUFFER_DATA(i) ((char *)0 + i)

namespace priv
{
    struct GlUtils
    {
        const int MY_GL_BITMAP = 1;
        const int REFRESH_TIME = 50;
    } glUtils;

    enum RenderingType {
        BUFFER_TO_TEXTURE,
        VBO,
        PBO
    };

    //VBO part
    GLuint vbo;
    struct cudaGraphicsResource* cuda_vbo_resource;
    void* d_vbo_buffer = NULL;

    //PBO, associated texture and shader
    GLuint gl_PBO, gl_Tex, gl_Shader;

    //handles OpenGL-CUDA exchange
    struct cudaGraphicsResource* cuda_pbo_resource; 
    
    //Source image on the host side
    Color* h_Src = 0;
    
    //Shows text information
    bool showInfos = true;

    //Type of rendering
    RenderingType renderingType = PBO;

    int myLastX;
    int myLastY;
    int myClickedButton = -1;
}



//const int windowX = 800;
//const int windowY = 800;
//float4* pixels = NULL;
//Color* onePassPixels = NULL;
//int pass = 0;

namespace raytracer {
    Scene* scene;
    Timer timer;
}

//#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
//// This is specifically to enable the application to enable/disable vsync
//typedef BOOL(WINAPI* PFNWGLSWAPINTERVALFARPROC)(int);
//
//void setVSync(int interval)
//{
//    if (WGL_EXT_swap_control)
//    {
//        wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");
//        wglSwapIntervalEXT(interval);
//    }
//}
//#endif


void DrawCircle(bool planehandle = true)
{
    int nside = 50;

    const double pi2 = 3.14159265 * 2.0;

    glBegin(GL_LINE_LOOP);

    for (double i = 0; i < nside; i++) {
        glNormal3d(cos(i * pi2 / nside), sin(i * pi2 / nside), 0.0);
        glVertex3d(cos(i * pi2 / nside), sin(i * pi2 / nside), 0.0);
    }

    glEnd();
}



/*!

  @brief Draw a spherical manipulator icon.



  @param tb the manipulator.

  @param active boolean to be set to true if the icon is active.

*/

void DrawSphereIcon(bool active, bool planeshandle = false)
{
    glPushAttrib(GL_TRANSFORM_BIT | GL_DEPTH_BUFFER_BIT | GL_ENABLE_BIT | GL_LINE_BIT | GL_CURRENT_BIT | GL_LIGHTING_BIT);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glDepthMask(GL_FALSE);

    glTranslatef(0.5f, 0.5f,1.f);
    glScalef(0.5f,0.5f,0.5f);

    float amb[4] = { .35f, .35f, .35f, 1.0f };
    float col[4] = { .5f, .5f, .8f, 1.0f };
    glEnable(GL_LINE_SMOOTH);
    if (active)
        glLineWidth(3);
   else
        glLineWidth(3);

    glDisable(GL_COLOR_MATERIAL); // has to be disabled, it is used by wrapper to draw meshes, and prevent direct material setting, used here

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor3f(0.5f,0.5f,0.5f);

    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, amb);


    col[0] = .40f; col[1] = .40f; col[2] = .85f;
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, col);

    DrawCircle(planeshandle);

    glRotatef(90, 1, 0, 0);
    col[0] = .40f; col[1] = .85f; col[2] = .40f;
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, col);
    DrawCircle(planeshandle);

    glRotatef(90, 0, 1, 0);
    col[0] = .85f; col[1] = .40f; col[2] = .40f;
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, col);
    DrawCircle(planeshandle);

    glPopMatrix();
    glPopAttrib();

}



//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void refreshTimer(int)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(priv::glUtils.REFRESH_TIME, refreshTimer, 0);
    }
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
// gl_Shader for displaying floating-point texture
static const char* shader_code =
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
GLuint compileASMShader(GLenum program_type, const char* code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte*)code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte* error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void drawText(float x, float y, void* font, const unsigned char* string, float3 rgb)
{
    glColor3f(rgb.x, rgb.y, rgb.z);
    glRasterPos2f(x, y);

    glutBitmapString(font, string);
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
inline void glPrint(int x, int y, const char* s, void* font)
{
    glRasterPos2f((GLfloat)x, (GLfloat)y);
    int len = (int)strlen(s);

    for (int i = 0; i < len; i++)
    {
        glutBitmapCharacter(font, s[i]);
    }
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
inline void glPrintShadowed(int x, int y, const char* s, void* font, float* color)
{
    glColor3f(0.0, 0.0, 0.0);
    glPrint(x - 1, y - 1, s, font);

    glColor3fv((GLfloat*)color);
    glPrint(x, y, s, font);
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
inline void beginWinCoords(void)
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glTranslatef(0.0, (GLfloat)(glutGet(GLUT_WINDOW_HEIGHT) - 1.0), 0.0);
    glScalef(1.0, -1.0, 1.0);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT), -1, 1);

    glMatrixMode(GL_MODELVIEW);
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
inline void endWinCoords(void)
{
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void drawText(float x, float y, const char* string, float* rgb, bool blend, void* font = nullptr)
{
    glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
    if(blend)
        glEnable(GL_BLEND);
    beginWinCoords();
    if (!font)
        font = (void*)GLUT_BITMAP_TIMES_ROMAN_24;
    glPrintShadowed(x, y, string, font, rgb);
    endWinCoords();
    if(blend)
        glDisable(GL_BLEND);
    glColor3f(1, 1, 1);
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void mapVBO()
{
    // Vbo
    //VboVec3r* buffer = raytracer::scene->vbo();
    //cudaGLMapBufferObject((void**)&buffer, priv::vbo); // maps a buffer object for access by CUDA

    //glClear(GL_COLOR_BUFFER_BIT); //clear all pixels
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void drawVBO()
{
    //cudaGLUnmapBufferObject(priv::vbo);
    //glFlush();
    //glFinish();
    //glBindBuffer(GL_ARRAY_BUFFER, priv::vbo);
    //glVertexPointer(2, GL_FLOAT, 12, 0);
    //glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
    //glEnableClientState(GL_VERTEX_ARRAY);
    //glEnableClientState(GL_COLOR_ARRAY);
    //glDrawArrays(GL_POINTS, 0, size.x() * size.x());
    //glDisableClientState(GL_VERTEX_ARRAY);
    //glutSwapBuffers();
    //

    //Vbo 2
    // map OpenGL buffer object for writing from CUDA
    //float4* dptr;
    //cudaGraphicsMapResources(1, vbo_resource, 0);
    //size_t num_bytes;
    //checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
    //*vbo_resource));
    //glClear(GL_COLOR_BUFFER_BIT); //clear all pixels
    //raytracer::timer.reset();
    //raytracer::scene->render();
    //double elapsed = raytracer::timer.elapsedTime();
    //Screen& screen = raytracer::scene->getScreen();
    //const Point2i& size = screen.getSize();
    //// unmap buffer object
    //checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
    //
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void mapPBO()
{
    CUDA_SAFE(cudaGraphicsMapResources(1, &priv::cuda_pbo_resource, 0));
    size_t num_bytes;
    Color** pboBuffer = raytracer::scene->pbo();
    CUDA_SAFE(cudaGraphicsResourceGetMappedPointer((void**)pboBuffer, &num_bytes, priv::cuda_pbo_resource));
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void drawPBO()
{
    Screen& screen = raytracer::scene->getScreen();
    const Point2i& size = screen.getSize();
    CUDA_SAFE(cudaGraphicsUnmapResources(1, &priv::cuda_pbo_resource, 0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, priv::gl_PBO);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, priv::gl_Tex);
    //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.x(), size.y(), GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.x(), size.y(), GL_RGBA, GL_FLOAT, BUFFER_DATA(0));
   // glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, priv::gl_Shader);
  //  glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
  //  glDisable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_TEXTURE_2D);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void drawBufferToGlTexture()
{
    Screen& screen = raytracer::scene->getScreen();
    const Point2i& size = screen.getSize();
    Color* data = screen.data();
    int pass = data[0].w();
   //render(windowX, windowY, pixels, onePassPixels, 0, pass);
   //pass++;
   

   uchar3* image = new uchar3[uint64_t(size.x()) * uint64_t(size.y())];

   for (int i = 0; i < size.x() * size.y(); i++) {
       image[i] = make_uchar3(math::clamp((int)(data[i].x() * 255. / pass), 0, 255),
           math::clamp((int)(data[i].y() * 255. / pass), 0, 255),
           math::clamp((int)(data[i].z() * 255. / pass), 0, 255));

       //image[i] = make_uchar3(math::clamp((int)(pixels[i].x * 255. / pass), 0, 255), 
       //                       math::clamp((int)(pixels[i].y * 255. / pass), 0, 255), 
       //                       math::clamp((int)(pixels[i].z * 255. / pass), 0, 255));
   }

   glEnable(GL_TEXTURE_2D);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size.x(), size.y(), 0, GL_RGB, GL_UNSIGNED_BYTE, image);
   //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowX, windowY, 0, GL_RGBA, GL_FLOAT, (uint8_t*) pixels);

   delete[] image;

   // Show the texture:

   glBindTexture(GL_TEXTURE_2D, priv::glUtils.MY_GL_BITMAP);
   glBegin(GL_QUADS);
   glTexCoord2f(0.0, 0.0);
   glVertex3f(0.0, 1.0, 0.0);
   glTexCoord2f(1.0, 0.0);
   glVertex3f(1.0, 1.0, 0.0);
   glTexCoord2f(1.0, 1.0);
   glVertex3f(1.0, 0.0, 0.0);
   glTexCoord2f(0.0, 1.0);
   glVertex3f(0.0, 0.0, 0.0);
   glEnd();
   glDisable(GL_TEXTURE_2D);
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void drawTextInfo(double elapsed)
{
    std::string textIter = "Iterations: " + tools::numToString(raytracer::scene->getNumberIterations()) 
        + " (samples: " + tools::numToString(raytracer::scene->getNumberSamplesDone()) + ")";
    float color[] = { 1.f, 1.f, 1.f };
    drawText(0, 24, textIter.c_str(), color, false);

    textIter = "fps: " + tools::numToString(1. / elapsed);
    drawText(0, 48, textIter.c_str(), color, false);
}

// OpenGL display function
//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void displayFunc(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //
    //Buffer to texture mode
    //raytracer::timer.reset();
    //raytracer::scene->render();
    //double elapsed = raytracer::timer.elapsedTime();
    //drawBufferToGlTexture();

    //std::string textIter = "Iterations: " + tools::numToString(raytracer::scene->getNumberIterations()) + " (samples: " + tools::numToString(raytracer::scene->getNumberSamplesDone()) + ")";
    //float color[] = { 1.f, 1.f, 1.f };
    //drawText(0, 24, textIter.c_str(), color, false);
    ////glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
    ////glEnable(GL_BLEND);
    ////beginWinCoords();
    ////float col[3] = { 1.f, 1.f, 1.f };
    ////char textIter[100];
    ////sprintf(textIter, "Iterations: %u", pass);
    ////glPrintShadowed(0, 24, textIter, (void*)GLUT_BITMAP_TIMES_ROMAN_24, col);
    ////endWinCoords();
    ////glDisable(GL_BLEND);
    ////glColor3f(1, 1, 1);

    //textIter = "fps: " + tools::numToString(1. / elapsed);
    //drawText(0, 48, textIter.c_str(), color, false);
    ////glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
    ////glEnable(GL_BLEND);
    ////beginWinCoords();
    ////float col[3] = { 1.f, 1.f, 1.f };
    ////char textIter[100];
    ////sprintf(textIter, "fps: %f", 1. / elapsed);
    ////glPrintShadowed(0, 48, textIter, (void*)GLUT_BITMAP_TIMES_ROMAN_24, col);
    ////endWinCoords();
    ////glDisable(GL_BLEND);
    ////glColor3f(1, 1, 1);

    ////glDisable(GL_DEPTH_TEST);
    ////glColor4f(0.0, 0.0, 0.0, 0.0);
    ////char textIter[100];
    ////sprintf(textIter, "Iterations: %u", pass);
    ////glRasterPos2f(0.01, 0.972);
    ////for (unsigned int i = 0; i < strlen(textIter); i++) {
    ////    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, textIter[i]);
    ////}
    ////glColor4f(1.0, 1.0, 1.0, 1.0);
    ////glEnable(GL_DEPTH_TEST);

    ////glPushMatrix();
    ////glTranslatef(0.1,0.2, 0.0);
    ////for (int i = 0; i < strlen(textIter); i++)
    ////{
    ////    glColor3f(1,1, 0.0);
    ////    glutStrokeCharacter(GLUT_STROKE_ROMAN, textIter[i]);
    ////}
    ////glPopMatrix();
    ////glColor3f(1, 1, 1);

    //glutSwapBuffers();
    ////glutPostRedisplay();

    if(priv::renderingType == priv::VBO)
        mapVBO();
    else if(priv::renderingType == priv::PBO)
        mapPBO();

    //compute
    raytracer::timer.reset();
    raytracer::scene->render();
    double elapsed = raytracer::timer.elapsedTime();

    if (priv::renderingType == priv::BUFFER_TO_TEXTURE)
        drawBufferToGlTexture();
    if (priv::renderingType == priv::VBO)
        drawVBO();
    else if (priv::renderingType == priv::PBO)
        drawPBO();

    if(priv::showInfos)
        drawTextInfo(elapsed);

   // DrawSphereIcon(true);

    glutSwapBuffers();
    glutPostRedisplay();
    
} // displayFunc

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void cleanup()
{
    if (priv::h_Src)
    {
        free(priv::h_Src);
        priv::h_Src = 0;
    }

    if (priv::renderingType == priv::PBO) {
        CUDA_SAFE(cudaGraphicsUnregisterResource(priv::cuda_pbo_resource));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        glDeleteBuffers(1, &priv::gl_PBO);
        glDeleteTextures(1, &priv::gl_Tex);
    }
    //glDeleteProgramsARB(1, &gl_Shader);

    delete raytracer::scene;
}

// OpenGL keyboard function
//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void keyboardFunc(unsigned char k, int, int)
{
    switch (k)
    {
    case '\033':
        printf("Shutting down...\n");

#if defined(__APPLE__) || defined(MACOSX)
        exit(EXIT_SUCCESS);
#else
        glutDestroyWindow(glutGetWindow());
        return;
#endif
        break;

    case 'z': {
        Point2i size = raytracer::scene->getScreen().getSize();
        const Transform& transform = TrackballHelper::forwardBackward(size, 1.f); //set 2s to go through the whole scene bbox
        raytracer::scene->setCameraWorldTransform(transform);
        break;
    }

    case 's': {
        Point2i size = raytracer::scene->getScreen().getSize();
        const Transform& transform = TrackballHelper::forwardBackward(size, -1.f); //set 2s to go through the whole scene bbox
        raytracer::scene->setCameraWorldTransform(transform);
        break;
    }

    case 'P': {
        int spp = raytracer::scene->getSamplesPerPixel();
        if (spp > 4) {
            spp /= 2;
            raytracer::scene->setSamplesPerPixel(spp);
        }
        break;
    }

    case 'p': {
        int spp = raytracer::scene->getSamplesPerPixel();
        spp *= 2;
        raytracer::scene->setSamplesPerPixel(spp);
        break;
    }

    case 'q':
        //raytracer::scene->moveCamera(Point3r(-1, 0, 0));
        break;

    case 'd':
        //raytracer::scene->moveCamera(Point3r(1, 0, 0));
        break;

    case 'r':
        //if (priv::renderingType == priv::PBO)
        //    std::fill(priv::h_Src, priv::h_Src + (uint64_t(800) * 800), Color(real(0)));
        raytracer::scene->reset();     
        break;

    case 'i':
        priv::showInfos = !priv::showInfos;
        break;

    default:
        break;
    }

} // keyboardFunc

// OpenGL mouse click function
//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void clickFunc(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        priv::myClickedButton = button;
    }
    else
        priv::myClickedButton = -1;

    //int modifiers = glutGetModifiers();

    //if (leftClicked && (modifiers & GLUT_ACTIVE_SHIFT))
    //{
    //    leftClicked = 0;
    //    middleClicked = 1;
    //}

    priv::myLastX = x;
    priv::myLastY = y;
} // clickFunc


// OpenGL mouse motion function
//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void motionFunc(int x, int y)
{
    Point2i size = raytracer::scene->getScreen().getSize();

    if (priv::myClickedButton == GLUT_LEFT_BUTTON)
    {
        const Transform& transform = TrackballHelper::sphereMode(size, Point2i(x, y), Point2i(priv::myLastX, priv::myLastY));
        raytracer::scene->setCameraWorldTransform(transform);
    } 
    else if (priv::myClickedButton == GLUT_MIDDLE_BUTTON)
    {
        const Transform& transform = TrackballHelper::pan(size, Point2i(x, y), Point2i(priv::myLastX, priv::myLastY));
        raytracer::scene->setCameraWorldTransform(transform);
    }

    priv::myLastX = x;
    priv::myLastY = y;
} // motionFunc

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void mouseWheel(int button, int dir, int x, int y)
{
    Point2i size = raytracer::scene->getScreen().getSize();
    const Transform& transform = TrackballHelper::zoom(size, Point2i(x, y), Point2i(priv::myLastX, priv::myLastY), dir);
    raytracer::scene->setCameraWorldTransform(transform);
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void initBufferToTexture()
{
    // Create a texture for displaying the render:
    glBindTexture(GL_TEXTURE_2D, priv::glUtils.MY_GL_BITMAP);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // Use nearest-neighbor point sampling instead of linear interpolation:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    // Enable textures:
    glEnable(GL_TEXTURE_2D);
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void initVBO(int sizeX, int sizeY)
{
    //VBO
    //Create vertex buffer object
    //glGenBuffers(1, &(priv::vbo));
    //glBindBuffer(GL_ARRAY_BUFFER, priv::vbo);
    ////Initialize VBO
    //unsigned int size = sizeX * sizeY * sizeof(Point3r);
    //glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    //glBindBuffer(GL_ARRAY_BUFFER, 0);
    ////Register VBO with CUDA
    //cudaGLRegisterBufferObject(priv::vbo);
    ////

    ////VBO 2
    //// create buffer object
    //glGenBuffers(1, &(priv::vbo));
    //glBindBuffer(GL_ARRAY_BUFFER, priv::vbo);

    //// initialize buffer object
    //unsigned int size = sizeX * sizeY * 4 * sizeof(float);
    //glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    //glBindBuffer(GL_ARRAY_BUFFER, 0);
    //// register this buffer object with CUDA
    //cudaGraphicsGLRegisterBuffer(&priv::cuda_vbo_resource, priv::vbo, cudaGraphicsMapFlagsWriteDiscard);
    //
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void initPBO(int sizeX, int sizeY)
{
    priv::h_Src = (Color*)malloc(uint64_t(sizeX) * sizeY * sizeof(Color));
    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &priv::gl_Tex);
    glBindTexture(GL_TEXTURE_2D, priv::gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, sizeX, sizeY, 0, GL_RGBA, GL_UNSIGNED_BYTE, priv::h_Src);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, sizeX, sizeY, 0, GL_RGBA, GL_FLOAT, priv::h_Src);
    printf("Texture created.\n");
    glDisable(GL_TEXTURE_2D);

    printf("Creating PBO...\n");
    glGenBuffers(1, &priv::gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, priv::gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, uint64_t(sizeX) * sizeY * sizeof(Color), priv::h_Src, GL_STREAM_COPY);
    //While a PBO is registered to CUDA, it can't be used
    //as the destination for OpenGL drawing calls.
    //But in our particular case OpenGL is only used
    //to display the content of the PBO, specified by CUDA kernels,
    //so we need to register/unregister it only once.
    cudaGraphicsGLRegisterBuffer(&priv::cuda_pbo_resource, priv::gl_PBO,
        cudaGraphicsMapFlagsWriteDiscard);

    priv::gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
void initOpenGL(int& argc, char** argv, int windowX, int windowY)
{ 
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(windowX, windowY);
    glutInitWindowPosition(0, 0);
    glutCreateWindow(argv[0]);

    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
    glutMouseWheelFunc(mouseWheel);
    glutMouseFunc(clickFunc);
    glutMotionFunc(motionFunc);
    //glutReshapeFunc(reshapeFunc);
    //glutTimerFunc(priv::glUtils.REFRESH_TIME, refreshTimer, 0);

    //Initialize necessary OpenGL extensions
    glewInit();
    if (!isGLVersionSupported(1, 5) ||
        !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_SUCCESS);
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    //Viewport
    glViewport(0, 0, windowX, windowY);

    //Orthographic view:
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    //Texture mode
    if(priv::renderingType == priv::BUFFER_TO_TEXTURE)
        initBufferToTexture();
    else if(priv::renderingType == priv::VBO) //VBO mode
        initVBO(windowX, windowY);
    else //PBO mode
        initPBO(windowX, windowY);
}

#include "core/geometry.h"

//=============================================================================
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    // check for hardware double precision support
    int dev = 0;
    dev = cutils::findCudaDevice();

    cudaDeviceProp deviceProp;
    CUDA_SAFE(cudaGetDeviceProperties(&deviceProp, dev));
    int runtimeVersion;// = deviceProp.major * 10 + deviceProp.minor;
    int driverVersion;
    CUDA_SAFE(cudaRuntimeGetVersion(&runtimeVersion));
    CUDA_SAFE(cudaDriverGetVersion(&driverVersion));

    int numSMs = deviceProp.multiProcessorCount;

    std::cout << "CUDA. Runtime version: " << runtimeVersion << std::endl << "Driver version: " << driverVersion << std::endl << "Num SM: " << numSMs << std::endl;

    priv::showInfos = true;

    bool allocateMemory = !(priv::renderingType == priv::PBO || priv::renderingType == priv::VBO);
    raytracer::scene = new Scene(allocateMemory);
    const Point2i& size = raytracer::scene->getScreen().getSize();
    //const int windowX = 800;
    //const int windowY = 800;
    printf("Initializing GLUT...\n");
    initOpenGL(argc, argv, size.x(), size.y());

    // Otherwise it succeeds, we will continue to run this sample
    //initData(argc, argv);

    // Initialize OpenGL context first before the CUDA context is created.  This is needed
    // to achieve optimal performance with OpenGL/CUDA interop.
    //initOpenGLBuffers(imageW, imageH);

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    //setVSync(0);
#endif

    //pixels = new float4[windowX * windowY];
    //std::fill(pixels, pixels + (windowX * windowY), make_float4(0, 0, 0, 0));
    //onePassPixels = new Color[windowX * windowY];

    //std::thread th([&]() {
    //    while (true) {
    //        render(windowX, windowY, pixels, onePassPixels, 0, pass);
    //        pass++;
    //    }
    //    }
    //);

    //render(windowX, windowY, pixels, 0);// &camera);
    glutMainLoop();

    return 0;
}









//#include <GL/freeglut.h>
//#include <iostream>
//#include <fstream>
//
//GLfloat UpwardsScrollVelocity = -10.0;
//float view = 20.0;
//
//char quote[6][80];
//int numberOfQuotes = 0, i;
//
////*********************************************
////*  glutIdleFunc(timeTick);                  *
////*********************************************
//
//void timeTick(void)
//{
//    if (UpwardsScrollVelocity < -600)
//        view -= 0.000011;
//    if (view < 0) { view = 20; UpwardsScrollVelocity = -10.0; }
//    //  exit(0);
//    UpwardsScrollVelocity -= 0.015;
//    glutPostRedisplay();
//
//}
//
//
////*********************************************
////* printToConsoleWindow()                *
////*********************************************
//
//void printToConsoleWindow()
//{
//    int l, lenghOfQuote, i;
//
//    for (l = 0; l < numberOfQuotes; l++)
//    {
//        lenghOfQuote = (int)strlen(quote[l]);
//
//        for (i = 0; i < lenghOfQuote; i++)
//        {
//            //cout<<quote[l][i];
//        }
//        //out<<endl;
//    }
//
//}
//
////*********************************************
////* RenderToDisplay()                       *
////*********************************************
//
//void RenderToDisplay()
//{
//    int l, lenghOfQuote, i;
//
//    glTranslatef(0.0, -100, UpwardsScrollVelocity);
//    glRotatef(-20, 1.0, 0.0, 0.0);
//    glScalef(0.1, 0.1, 0.1);
//
//
//
//    for (l = 0; l < numberOfQuotes; l++)
//    {
//        lenghOfQuote = (int)strlen(quote[l]);
//        glPushMatrix();
//        glTranslatef(-(lenghOfQuote * 37), -(l * 200), 0.0);
//        for (i = 0; i < lenghOfQuote; i++)
//        {
//            glColor3f((UpwardsScrollVelocity / 10) + 300 + (l * 10), (UpwardsScrollVelocity / 10) + 300 + (l * 10), 0.0);
//            glutStrokeCharacter(GLUT_STROKE_ROMAN, quote[l][i]);
//        }
//        glPopMatrix();
//    }
//
//}
////*********************************************
////* glutDisplayFunc(myDisplayFunction);       *
////*********************************************
//
//void myDisplayFunction(void)
//{
//    glClear(GL_COLOR_BUFFER_BIT);
//    glLoadIdentity();
//    gluLookAt(0.0, 30.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
//    RenderToDisplay();
//    glutSwapBuffers();
//}
////*********************************************
////* glutReshapeFunc(reshape);               *
////*********************************************
//
//void reshape(int w, int h)
//{
//    glViewport(0, 0, w, h);
//    glMatrixMode(GL_PROJECTION);
//    glLoadIdentity();
//    gluPerspective(60, 1.0, 1.0, 3200);
//    glMatrixMode(GL_MODELVIEW);
//}
//
////*********************************************
////* int main()                                *
////*********************************************
//
//
//int main(int argc, char* argv[])
//{
//    strcpy(quote[0], "Luke, I am your father!.");
//    strcpy(quote[1], "Obi-Wan has taught you well. ");
//    strcpy(quote[2], "The force is strong with this one. ");
//    strcpy(quote[3], "Alert all commands. Calculate every possible destination along their last known trajectory. ");
//    strcpy(quote[4], "The force is with you, young Skywalker, but you are not a Jedi yet.");
//    numberOfQuotes = 5;
//
//    glutInit(&argc, argv);
//    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
//    glutInitWindowSize(800, 400);
//    glutCreateWindow("StarWars scroller");
//    glClearColor(0.0, 0.0, 0.0, 1.0);
//    glLineWidth(3);
//
//    glutDisplayFunc(myDisplayFunction);
//    glutReshapeFunc(reshape);
//    glutIdleFunc(timeTick);
//    glutMainLoop();
//
//    return 0;
//}