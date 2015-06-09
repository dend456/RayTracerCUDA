#pragma warning (disable : 4996)

#include "Vector.h"
#include "Quaternion.h"
#include "Camera.h"
#include "structs.h"

#include <vector>
#include <map>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fstream>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
// includes, cuda
#include "cuda.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#include "FEClock.h"
#include "lodepng.h"
#include <random>

using namespace std;


#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY    100 //ms
#define MAX_DEPTH 3

#define E 0.00001f

#define PI 3.14159265358979f
#define PIOVER180 0.01745329251f

#define WRAP 1
#define CLAMP 2
#define MIRROR 3

const unsigned int window_width  = 512;
const unsigned int window_height = 512;
const unsigned int max_shapes = 2048;
const unsigned int max_lights = 64;


const unsigned int sphere = 1;
const unsigned int box = 2;


float fovx = PI / 4;
float fovy = (window_height / window_width) * fovx;

__device__ __host__ __inline__ float degreesToRadians(float a)
{
	return a * PIOVER180;
}



////////////////////////////////////////////////////////////////////////////////
// constants


GLfloat colors[window_height * window_width * 3];
float* devColors;
Segment rays[window_height * window_width];
Segment* devRays;
Shape shapes[max_shapes];
Shape* devShapes;
int numShapes = 0;
Light lights[max_lights];
Light* devLights;
int numLights = 0;
std::map<const char*, Image> loadedImages;

bool lockFPS = false;
float timePerFrame = 1.0f/60.0f;

int antialiasing = 2;

FEClock* feclock;

GLuint pb;
void* cpb;

Camera camera;
Vector center(0,0,0);

GLuint screenRect = 0;

unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

void cleanup();

bool initGL(int *argc, char **argv);

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);

void runCuda();
void loadScene(string file);


__device__ bool intersects(Segment* s, Shape* shape, Vector* outPos, Vector* norm, float* t);

__device__ __inline__ float clamp(float f, float mini, float maxi)
{
	if(f < mini)
	{
		return mini;
	}
	if(f > maxi)
	{
		return maxi;
	}
	return f;
}

__device__ bool boxIntersects(Segment* s, Shape* shape, Vector* outPos, Vector* norm, float* t)
{
	Vector minimum(shape->pos.x - shape->bounds.x,shape->pos.y - shape->bounds.y,shape->pos.z - shape->bounds.z);
	Vector maximum(shape->pos.x + shape->bounds.x,shape->pos.y + shape->bounds.y,shape->pos.z + shape->bounds.z);

	float tmin = FLT_MIN;
	float tmax = FLT_MAX;
	float pi, di;
	float mini, maxi;

	for(int i=0;i<3;i++){
		if(i == 0){
			pi = s->pos.x;
			di = s->dir.x;
			mini = minimum.x;
			maxi = maximum.x;
		}
		else if(i == 1){
			pi = s->pos.y;
			di = s->dir.y;
			mini = minimum.y;
			maxi = maximum.y;
		}
		else{
			pi = s->pos.z;
			di = s->dir.z;
			mini = minimum.z;
			maxi = maximum.z;
		}

		if(abs(di) < E){
			if(pi < mini || pi > maxi){
				return false;
			}
		}
		else{
			float ood = 1.0f / di;
			float t1 = (mini - pi) * ood;
			float t2 = (maxi - pi) * ood;

			if(t1 > t2){
				float tmp = t1;
				t1 = t2;
				t2 = tmp;
			}

			tmin = (tmin > t1) ? tmin : t1;
			tmax = (tmax > t2) ? t2 : tmax;

			if(tmin > tmax){
				return false;
			}
		}
	}

	*t = tmin;
	s->dir.copy(outPos);
	outPos->mul(tmin);
	outPos->add(s->pos);


	if(abs(outPos->x - minimum.x) < E){
		norm->x = -1.0f;
		norm->y = 0.0f;
		norm->z = 0.0f;
	}
	else if(abs(outPos->x - maximum.x) < E){
		norm->x = 1.0f;
		norm->y = 0.0f;
		norm->z = 0.0f;
	}
	else if(abs(outPos->y - minimum.y) < E){
		norm->x = 0.0f;
		norm->y = -1.0f;
		norm->z = 0.0f;
	}
	else if(abs(outPos->y - maximum.y) < E){
		norm->x = 0.0f;
		norm->y = 1.0f;
		norm->z = 0.0f;
	}
	else if(abs(outPos->z - minimum.z) < E){
		norm->x = 0.0f;
		norm->y = 0.0f;
		norm->z = -1.0f;
	}
	else if(abs(outPos->z - maximum.z) < E){
		norm->x = 0.0f;
		norm->y = 0.0f;
		norm->z = 1.0f;
	}

	return true;
}

__device__ bool sphereIntersects(Segment* s, Shape* shape, Vector* outPos, Vector* norm, float* t)
{
	Vector dir = s->dir;

	Vector m = s->pos;
	m.sub(shape->pos);

	float b = m.dot(dir);
	float c = m.dot(m) - (shape->bounds.x * shape->bounds.x);

	if(c > 0.0f && b > 0.0f){
		return false;
	}

	float disc = b * b - c;

	if(disc < 0.0f){
		return false;
	}

	*t = -b - sqrt(disc);

	if(*t < 0.0f){
		*t = 0.0f;
	}
	
	s->pos.copy(outPos);
	dir.mul(*t);

	outPos->add(dir);

	outPos->copy(norm);
	norm->sub(shape->pos);
	norm->normalize();

	return true;
}

__device__ bool inShadow(Vector* point, Light* light, float distance, Shape* shapes, int numShapes)
{
	Segment ray;
	point->copy(&ray.pos);
	light->pos.copy(&ray.dir);
	ray.dir.sub(*point);
	ray.dir.normalize();
	
	Vector smallDist;
	ray.dir.copy(&smallDist);
	smallDist.mul(.1f);
	ray.pos.add(smallDist);

	Vector out;
	Vector norm;
	float t;

	for(int i=0;i<numShapes;i++){
		if(intersects(&ray,&shapes[i],&out,&norm,&t)){
			if(t < distance)
			{
				return true;
			}
		}
	}
	
	return false;
}
__device__ bool intersects(Segment* s, Shape* shape, Vector* outPos, Vector* norm, float* t)
{
	switch(shape->type)
	{
		case sphere:
			return sphereIntersects(s, shape, outPos, norm, t);

		case box:
			return boxIntersects(s, shape, outPos, norm, t);

		default:
			return false;
	}
}

__device__ int getTexel(Image& image, float u, float v)
{
	if(u > 1.0f)
	{
		if(image.wrapMode == WRAP)
		{
			float garb;
			u = modf(u, &garb);
		}
		else if(image.wrapMode == CLAMP)
		{
			u = 1.0f;
		}
		else if(image.wrapMode == MIRROR)
		{
			float mirror;
			u = modf(u, &mirror);
			if((int)mirror % 2)
			{
				u = 1.0f - u;
			}
		}
	}
	if(v > 1.0f)
	{
		if(image.wrapMode == WRAP)
		{
			float garb;
			v = modf(v, &garb);
		}
		else if(image.wrapMode == CLAMP)
		{
			v = 1.0f;
		}
		else if(image.wrapMode == MIRROR)
		{
			float mirror;
			v = modf(v, &mirror);
			if((int)mirror % 2)
			{
				v = 1.0f - v;
			}
		}
	}

	int texX = -image.width * u;
	int texY = image.height * v;

	if(texX >= image.width)
	{
		texX = image.width - 1;
	}
	if(texY >= image.height)
	{
		texY = image.height - 1;
	}

	return ((texY * image.width) + texX) * 4;

}

__device__ void getTextureData(Image& image, int texel, float& x, float& y, float& z, float& w)
{
	float* textureData = image.data;
	x = textureData[texel];
	y = textureData[texel+1];
	z = textureData[texel+2];
	w = textureData[texel+3];
}

__device__ Vector traceRay(Segment* ray, Shape* shapes, Light* lights, int numShapes, int numLights, int depth = 0)
{
	Vector color(0,0,0);

	Vector minNorm;
	Vector minPos;
	float minTime = 99999;
	int minShape = -1;
	Vector norm;
	Vector pos;
	float time;

	bool useTexture = false;
	bool useNormalMap = false;
	bool useSpecularMap = false;

	int texel = -1;

	for(int i=0;i<numShapes;++i)
	{	
		if(intersects(ray,&shapes[i],&pos,&norm,&time))
		{
			if(time < minTime)
			{
				minTime = time;
				norm.copy(&minNorm);
				pos.copy(&minPos);

				minShape = i;
			}
		}
	}

	if(minShape >= 0)
	{
		Vector tmpVec;
		Shape& shape = shapes[minShape];
		if(shape.type == sphere)
		{
			Image* texture = &shape.texture;
			if(shape.texture.data != NULL)
			{
				useTexture = true;
			}
			if(shape.normalMap.data != NULL)
			{
				useNormalMap = true;
				texture = &shape.normalMap;
			}
			if(shape.specularMap.data != NULL)
			{
				useSpecularMap = true;
				texture = &shape.specularMap;
			}
			if(useTexture || useNormalMap || useSpecularMap)
			{
				Vector rotPos = minPos;
				rotPos.sub(shape.pos);
				Vector newPos = shape.rotation.mul(rotPos);
				newPos.add(shape.pos);

				Vector d = shape.pos;
				d.sub(newPos);
				d.normalize();
				
				float u = 0.5f + (atan2(d.z,d.x) / (2*PI));
				float v = 0.5f - (asin(d.y) / PI);

				texel = getTexel(*texture, u, v);
			}
			if(useNormalMap)
			{
				Vector t;
				float garbage;
				getTextureData(shape.normalMap,texel,t.x,t.y,t.z,garbage);

				t.x = t.x * 2.0f - 1.0f;
				t.y = t.y * 2.0f - 1.0f;
				t.z = -(t.z * 2.0f - 1.0f);
				t.normalize();

				Vector z1(minNorm.x, minNorm.y, 1.0f);
				Quaternion rotation;
				rotation = rotation.rotationBetweenVector(z1,t);
				rotation.normalize();
				rotation = rotation.inverse();
				t = rotation.mul(t);

				//t = shape.rotation.mul(t);
				//t.normalize();
				minNorm = t;

				minNorm.normalize();
				//minNorm = shape.rotation.mul(minNorm);
				//minNorm.normalize();
			}
		}

		for(int i=0; i<numLights; ++i)
		{
			Vector lightVec;
			lights[i].pos.copy(&lightVec);
			lightVec.sub(minPos);
			float distance = lightVec.length();

			if(!inShadow(&minPos, &lights[i], distance, shapes, numShapes))
			{

				lightVec.div(distance);

				float NdL = minNorm.dot(lightVec);

				float diffuse = max(NdL,0.0f);
				
				lights[i].diffuseColor.copy(&tmpVec);
				tmpVec.mul(diffuse);
				tmpVec.mul(shape.color);
				tmpVec.mul(lights[i].diffuse);
				color.add(tmpVec);

				Vector refl;
				minNorm.copy(&refl);
				refl.mul(2.0f * NdL);
				refl.sub(lightVec);

				if(shape.shininess > 0.0f)
				{
					Vector view(0,0,10);

					view.sub(minPos);
					view.normalize();

					float specular = 0.0f;
			
					if(diffuse > 0.0f)
					{
						specular = pow(max(refl.dot(view),0.0f),shape.shininess);
					}
				
					lights[i].specularColor.copy(&tmpVec);
					tmpVec.mul(specular);
					tmpVec.mul(shape.color);
					tmpVec.mul(lights[i].specular);

					if(useSpecularMap)
					{
						Vector t;
						float garbage;
						getTextureData(shape.specularMap,texel,t.x,t.y,t.z,garbage);
						tmpVec.mul(t);
					}

					color.add(tmpVec);
				}
			}
		
			lights[i].ambientColor.copy(&tmpVec);
			tmpVec.mul(lights[i].ambient);
			
			if(useTexture)
			{
				Vector t;
				float garbage;
				getTextureData(shape.texture,texel,t.x,t.y,t.z,garbage);
				tmpVec.mul(t);
				tmpVec.mul(shape.color);
			}
			else
			{
				tmpVec.mul(shape.color);
			}

			color.add(tmpVec);
		}
		
		if(shape.reflectivity > 0.0f){
			if(depth < MAX_DEPTH)
			{
				Vector reflected;
				float idotn = minNorm.dot(ray->dir) * 2;
				ray->dir.copy(&reflected);
			
				minNorm.copy(&tmpVec);
				tmpVec.mul(idotn);
				reflected.sub(tmpVec);
		
				Segment s;
				reflected.copy(&s.dir);
				minPos.copy(&s.pos);
			
				//Vector smallDist;
				reflected.copy(&tmpVec);
				tmpVec.mul(.01f);

				s.pos.add(tmpVec);
				s.dir.normalize();

				tmpVec = traceRay(&s,shapes, lights,numShapes,numLights,depth+1);

				tmpVec.mul(shape.reflectivity);
				color.add(tmpVec);
			}
		}
	}

	return color;
}


__global__ void trace(float* colors, Camera camera, Shape* gshapes, Light* lights, int width, int height, int numShapes, int numLights, int aa)
{
	extern __shared__ Shape sshapes[];
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= width * height) return;

	int x = i % width;
	int y = i / width;

	/*if(threadIdx.x == 0)
	{
		memcpy(sshapes,gshapes,sizeof(Shape)*numShapes);
	}
	*/

	int size = numShapes / numShapes;
	if(threadIdx.x < numShapes)
	{
		memcpy(&sshapes[threadIdx.x*size], &gshapes[threadIdx.x*size], size*sizeof(Shape));
	}

	__syncthreads();

	int index = i * 3;

	Vector color(0,0,0);

	for(int aax = 0; aax < aa; ++aax)
	{
		for(int aay = 0; aay < aa; ++aay)
		{
			Segment s;
			s.pos = camera.pos;
			float xoff, yoff;
			float aspectRatio = (float)width / height;

			if(aa == 1)
			{
				if(width > height)
				{
					xoff = ((x + 0.5f) / width) * aspectRatio - (((width - height)/(float)height)/2); 
					yoff = ((height - y) + 0.5f) / height;
				}
				else if(height > width)
				{
					xoff = (x + 0.5f) / width;
					yoff = (((height - y) + 0.5f) / height) / aspectRatio - (((height - width) / (float)width) / 2);
				}
				else
				{
					xoff = (x + 0.5f) / width;
					yoff = ((height - y) + 0.5f) / height;
				}
			}
			else
			{
				if(width > height)
				{
					xoff = ((x + (float)aax / ((float)aa - 1)) / width) * aspectRatio - (((width - height)/(float)height)/2); 
					yoff = ((height - y) + (float)aax / ((float)aa - 1)) / height;
				}
				else if(height > width)
				{
					xoff = (x + (float)aax / ((float)aa - 1)) / width;
					yoff = (((height - y) + (float)aax / ((float)aa - 1)) / height) / aspectRatio - (((height - width) / (float)width) / 2);
				}
				else
				{
					xoff = (x + (float)aax / ((float)aa - 1)) / width;
					yoff = ((height - y) + (float)aax / ((float)aa - 1)) / height;
				}
			}

			Vector cameraDirection = camera.lookat;
			cameraDirection.sub(camera.pos);
			cameraDirection.normalize();

			Vector right = camera.right;
			right.mul(xoff - 0.5);
	
			Vector down = camera.down;
			down.mul(yoff - 0.5);

			right.add(down);

			cameraDirection.add(right);
			cameraDirection.normalize();

			s.dir = cameraDirection;

			color.add(traceRay(&s,sshapes,lights,numShapes,numLights));
		}
	}
				Segment s;
			s.pos = camera.pos;
			float xoff, yoff;
			float aspectRatio = (float)width / height;

				if(width > height)
				{
					xoff = ((x + 0.5f) / width) * aspectRatio - (((width - height)/(float)height)/2); 
					yoff = ((height - y) + 0.5f) / height;
				}
				else if(height > width)
				{
					xoff = (x + 0.5f) / width;
					yoff = (((height - y) + 0.5f) / height) / aspectRatio - (((height - width) / (float)width) / 2);
				}
				else
				{
					xoff = (x + 0.5f) / width;
					yoff = ((height - y) + 0.5f) / height;
				}
			

			Vector cameraDirection = camera.lookat;
			cameraDirection.sub(camera.pos);
			cameraDirection.normalize();

			Vector right = camera.right;
			right.mul(xoff - 0.5);
	
			Vector down = camera.down;
			down.mul(yoff - 0.5);

			right.add(down);

			cameraDirection.add(right);
			cameraDirection.normalize();

			s.dir = cameraDirection;

			color.add(traceRay(&s,sshapes,lights,numShapes,numLights));
	color.div(aa*aa+1);

	//float u = (x % width) / (float)width;
	//float v = (x / width) / (float)height;
	//int tex = ((height  * v) * width) + (width * u);
	//tex *= 4;
	//uchar4 color = tex1Dfetch<uchar4>(earthTex, tex);
	 
	colors[index] = color.x;
	colors[index+1] = color.y;
	colors[index+2] = color.z;
}

__global__ void calculateRays(Segment* rays, Camera camera, Vector center, int width, int height)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int x = i % width;
	int y = i / width;

	if(x >= width || y >= height) return;

	Segment* s = &rays[(y * width) + x];
	/*
	float WW2 = width / 2.0f;
	float WH2 = height / 2.0f;


	Vector dist(camera.lookat.x - camera.pos.x, camera.lookat.y - camera.pos.y, camera.lookat.z - camera.pos.z);
	float len = dist.length();

	float xFrac = (float)(x - WW2) / (float)width;
	float yFrac = (float)(y - WH2) / (float)height;
	float zFrac = -((camera.pos.z - camera.lookat.z) / len) ;
//	float zFrac = -1.0f;


	s->dir.x = xFrac;
	s->dir.y = yFrac;
	s->dir.z = zFrac;

	s->dir.normalize();
	*/
	s->pos = camera.pos;
	float xoff, yoff;
	float aspectRatio = (float)width / height;

	if(width > height)
	{
		xoff = ((x + 0.5f) / width) * aspectRatio - (((width - height)/(float)height)/2); 
		yoff = ((height - y) + 0.5f) / height;
	}
	else if(height > width)
	{
		xoff = (x + 0.5f) / width;
		yoff = (((height - y) + 0.5f) / height) / aspectRatio - (((height - width) / (float)width) / 2);
	}
	else
	{
		xoff = (x + 0.5f) / width;
		yoff = ((height - y) + 0.5f) / height;
	}

	Vector cameraDirection = camera.lookat;
	cameraDirection.sub(camera.pos);
	cameraDirection.normalize();

	Vector right = camera.right;
	right.mul(xoff - 0.5);
	
	Vector down = camera.down;
	down.mul(yoff - 0.5);

	right.add(down);

	cameraDirection.add(right);
	cameraDirection.normalize();

	s->dir = cameraDirection;
}


void display()
{
	static int count = 0;
	count++;

	//printf("%d\n",count);

    glClear(GL_COLOR_BUFFER_BIT);

	cudaGLMapBufferObject(&cpb,pb);
    runCuda();
	cudaGLUnmapBufferObject(pb);

	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pb);
	glBindTexture( GL_TEXTURE_2D, screenRect); 

//	glTexSubImage2D(GL_TEXTURE_2D,0,0,0,window_width,window_height,GL_RGB,GL_FLOAT,colors);
	glTexSubImage2D(GL_TEXTURE_2D,0,0,0,window_width,window_height,GL_RGB,GL_FLOAT,0);

	glBegin( GL_QUADS );
		glTexCoord2d(0.0,0.0); glVertex2i(0,0);
		glTexCoord2d(1.0,0.0); glVertex2i(window_width,0);
		glTexCoord2d(1.0,1.0); glVertex2i(window_width,window_height);
		glTexCoord2d(0.0,1.0); glVertex2i(0,window_height);
	glEnd();

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
}

void keyboard(unsigned char key,int x, int y)
{
	switch(key){
		case 27:
			exit(0);
			break;
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7':
		case '8':
		case '9':
		case '0':
			{
				char str[32];
				sprintf(str,"scene%c.txt",key);
				loadScene(str);
				break;
			}

		case 'l':
			numLights ^= 1;
			break;
		case '=':
		case '+':
			antialiasing++;
			std::cout << antialiasing << std::endl;
			break;
		case '-':
		case '_':
			antialiasing--;
			std::cout << antialiasing << std::endl;
			break;
	}
}

void recalcRays(Camera& camera)
{
	int threadsPerBlock = 256;
    int blocksPerGrid =((window_width * window_height) + threadsPerBlock - 1) / threadsPerBlock;

	calculateRays<<<blocksPerGrid,threadsPerBlock>>>(devRays, camera, center, window_width,window_height);
}


void rotate(Shape* s, float angle, Vector& axis, Vector& origin){

		float x = s->pos.x - origin.x;
		float y = s->pos.y - origin.y;
		float z = s->pos.z - origin.z;

		float r = angle * PIOVER180;

		float sinr = sin(r);
		float cosr = cos(r);
		float omc = 1 - cosr;

		float z0 = cosr + ((axis.x * axis.x) * omc);
		float z1 = ((axis.x * axis.y) * omc) + (axis.z * sinr);
		float z2 = ((axis.x * axis.z) * omc) - (axis.y * sinr);

		float o0 = ((axis.x * axis.y) * omc) - (axis.z * sinr);
		float o1 = cosr + ((axis.y * axis.y) * omc);
		float o2 = ((axis.y * axis.z) * omc) + (axis.x * sinr);

		float t0 = ((axis.x * axis.z) * omc) + (axis.y * sinr);
		float t1 = ((axis.y * axis.z) * omc) - (axis.x * sinr);
		float t2 = cosr + ((axis.z * axis.z) * omc);

		float nx,ny,nz;

		nx = x * z0 + y * z1 + z * z2;
		ny = x * o0 + y * o1 + z * o2;
		nz = x * t0 + y * t1 + z * t2;

		s->pos.x = nx + origin.x;
		s->pos.y = ny + origin.y;
		s->pos.z = nz + origin.z;
}

void moveObjects(float dt)
{
	static int loopCount = 0;
	static float angle = 0;

	loopCount++;

	angle += dt;

	//Light* l = &lights[0];
	//l->pos.x = cos(angle) * 15;
	//l->pos.z = sin(angle) * 15;


	/*Vector origin(0.0f,0.0f,-4.0f);
	Vector axis(0.0f,1.0f,0.0f);

	for(int i=0; i<102;++i)
	{
		Shape* s = &shapes[i];
		rotate(s, -dt * 50, axis, origin);
	}*/

	Quaternion rot;
	//rot.rotateYawPitchRoll(dt*20,0.0f,0.0f);
	//shapes[0].rotation.mul(rot);
	//shapes[1].rotation.mul(rot);

	/*for(int i=0;i<1;i++)
	{
		//int x = rand() % 40 - 20;
		//int y = rand() % 40 - 20;
		//int z = rand() % 40 - 20;
		int x = 10;
		int y = 0;
		int z = 0;
		rot.rotateYawPitchRoll(dt*x,dt*y,dt*z);
		shapes[i].rotation.mul(rot);
	}*/

	for(int i=0;i<numShapes;++i)
	{
		Shape* s = shapes + i;
		Vector r = s->rotationAxis;
		r.mul(dt * s->rotationSpeed);
		rot.rotateYawPitchRoll(r.x,r.y,r.z);
		s->rotation.mul(rot);
		rotate(s,dt*s->speed,s->movementAxis,s->movementPoint);
	}
}

void update()
{
	static int count = 0;
	static float fpsdt = 0;

	FEClock frameTimer;
	float timeForFrame;

	frameTimer.update();
	float dt = feclock->update();
	fpsdt += dt;

	moveObjects(dt);

	display();

	timeForFrame = frameTimer.update();
	timePerFrame = 1.0f/30.f;

	lockFPS = false;

	if(lockFPS)
	{
		float diff = timePerFrame - timeForFrame;
		if(diff > 0)
		{
			Sleep((DWORD)(.95f * diff * 1000)); 
		}
		while(diff > 0)
		{
			timeForFrame += frameTimer.update();
			diff = timePerFrame - timeForFrame;
		}
	}

	count++;
	if(count % 50 == 0)
	{
		count = 0;
			
		char fps[16];
		sprintf(fps, "%3.03f",100.0f/fpsdt);
		glutSetWindowTitle(fps);
		fpsdt = 0.0f;
	}

	timeForFrame += frameTimer.update();
}

bool initGL(int argc, char **argv)
{

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda RT");
    glutDisplayFunc(display);
	glutIdleFunc(update);
   glutKeyboardFunc(keyboard);
   // glutMotionFunc(motion);


    glewInit();

    if (! glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }


    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_TEXTURE_2D);

	glGenTextures( 1, &screenRect);
	glBindTexture( GL_TEXTURE_2D, screenRect);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB8, window_width,window_height, GL_RGB, GL_FLOAT,colors);
	//glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F,window_width,window_height, 0, GL_RGB,GL_FLOAT, NULL);

	
	glGenBuffers(1,&pb);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pb);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, window_width*window_height*3*sizeof(float),NULL,GL_DYNAMIC_COPY);
	
	glBindTexture( GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER,0);

	glMatrixMode(GL_MODELVIEW);
	///glLoadIdentity();
    //glViewport(0, 0, window_width, window_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluOrtho2D(0,window_width, 0, window_height);
	
    SDK_CHECK_ERROR_GL();

    return true;
}

void cleanup()
{
	if(screenRect)
	{
		glDeleteTextures(1, &screenRect);
		screenRect = 0;
	}
	

	cudaGLUnmapBufferObject(pb);
	if(pb)
	{
		glDeleteBuffers(1,&pb);
		pb = 0;
	}

	delete feclock;

	//cudaFree(devColors);
	cudaFree(devShapes);
	//cudaFree(devRays);

	cudaDeviceReset();
}

void flipVerticle(byte* data, int width, int height)
{
	int bitsPerPixel = 32;
	int bytesPerPixel = bitsPerPixel / 8;
	int rowSize = width * bytesPerPixel;
	int size = height * rowSize;

	byte* dataCopy = new byte[size];

	memcpy(dataCopy, data, size);

	int bytesPerRow = width * bytesPerPixel;

	for(int i=height-1, k=0; i>=0; --i, ++k)
	{
		memcpy(&data[k * bytesPerRow], &dataCopy[i * bytesPerRow], rowSize);
	}

	delete dataCopy;
}

void loadTexture(const char* file, Image& texture, int wrapMode)
{
	if(loadedImages.find(file) != loadedImages.end())
	{
		Image i = loadedImages[file];
		texture.data = i.data;
		texture.height = i.height;
		texture.width = i.width;
		texture.wrapMode = wrapMode;
		return;
	}
	std::vector<unsigned char> image;
	unsigned int width;
	unsigned int height;
	unsigned int error = lodepng::decode(image, width, height, file);
	
	texture.width = width;
	texture.height = height;
	texture.wrapMode = wrapMode;
	
	int imageSize = width * height * 4;

	byte* data = new byte[imageSize];
	std::copy(image.begin(), image.end(), data);
	flipVerticle(data, width, height);

	int imageSizeFloat = width * height * 4 * sizeof(float);
	float* floatData = new float[imageSize];

	for(int i=0;i<imageSize;++i)
	{
		floatData[i] = data[i] / 255.0f;
	}
	
	cudaError_t err;
	err = cudaMalloc(&texture.data, imageSizeFloat);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to load texture", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(texture.data, floatData, imageSizeFloat, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy image data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	delete floatData;
	delete data;

	loadedImages[file] = texture;
}

void initCuda()
{
	cudaDeviceReset();
	cudaError_t err;
	/*err = cudaMalloc((void **)&devRays, window_width * window_height * sizeof(Segment));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device colors\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/
/*
	err = cudaMemcpy(devRays, rays, window_width * window_height * sizeof(Segment), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy rays from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/


	err = cudaMalloc((void **)&devShapes, max_shapes * sizeof(Shape));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device colors\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(devShapes, shapes, max_shapes * sizeof(Shape), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy shapes from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMalloc((void **)&devLights, max_lights * sizeof(Light));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device lights\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(devLights, lights, max_lights * sizeof(Light), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy lights from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	cudaGLRegisterBufferObject(pb);

	cudaGLMapBufferObject(&cpb, pb);

	err = cudaThreadSetLimit 	(cudaLimitStackSize,1024*2*2);

	size_t t;
	err = cudaThreadGetLimit 	( &t,cudaLimitStackSize);


	/////////////
}

void runCuda()
{	
	if(!cpb) return;
	
	int threadsPerBlock = 256;
	int blocksPerGrid =((window_width * window_height) + threadsPerBlock - 1) / threadsPerBlock;

	cudaMemcpy(devLights, lights, numLights * sizeof(Light), cudaMemcpyHostToDevice);
	cudaMemcpy(devShapes, shapes, numShapes * sizeof(Shape), cudaMemcpyHostToDevice);

	trace<<<blocksPerGrid,threadsPerBlock,sizeof(Shape)*numShapes>>>((float*)cpb,camera,devShapes,devLights,window_width,window_height,numShapes,numLights,antialiasing);
}

Vector fillVectorFromLine(string line)
{
	Vector v;
	istringstream ss(line);
	string token;
	char* end;

	getline(ss, token, ',');
	v.x = (float)strtod(token.c_str(),&end);
	getline(ss, token, ',');
	v.y = (float)strtod(token.c_str(),&end);
	getline(ss, token, ',');
	v.z = (float)strtod(token.c_str(),&end);

	return v;
}

int stringToWrap(string s)
{
	if(s == "clamp")
	{
		return CLAMP;
	}
	else if(s == "wrap")
	{
		return WRAP;
	}
	else if(s == "mirror")
	{
		return MIRROR;
	}
	else
	{
		return WRAP;
	}
}	

void loadSphere(ifstream& in, Shape* shape)
{
	string lines[21];
	char* end;

	shape->type = sphere;
	shape->texture.data = NULL;
	shape->normalMap.data = NULL;
	shape->bumpMap.data = NULL;
	shape->reflectionMap.data = NULL;

	for(int i=0;i<21;++i)
	{
		getline(in,lines[i]);
	}

	shape->shininess = strtol(lines[0].c_str(),&end,10);
	shape->reflectivity = (float)strtod(lines[1].c_str(),&end);
	shape->pos = fillVectorFromLine(lines[2]);
	shape->bounds = fillVectorFromLine(lines[3]);
	shape->color = fillVectorFromLine(lines[4]);
	if(lines[5] != "none")
	{
		loadTexture(lines[5].c_str(),shape->texture,stringToWrap(lines[6]));
	}
	if(lines[7] != "none")
	{
		loadTexture(lines[7].c_str(),shape->normalMap,stringToWrap(lines[8]));
	}
	if(lines[9] != "none")
	{
		loadTexture(lines[9].c_str(),shape->bumpMap,stringToWrap(lines[10]));
	}
	if(lines[11] != "none")
	{
		loadTexture(lines[11].c_str(),shape->reflectionMap,stringToWrap(lines[12]));
	}
	if(lines[13] != "none")
	{
		loadTexture(lines[13].c_str(),shape->specularMap,stringToWrap(lines[13]));
	}
	shape->rotation.set(0,0,0,1);
	Vector rot = fillVectorFromLine(lines[15]);
	shape->rotation.rotateYawPitchRoll(rot.x,rot.y,rot.z);
	shape->rotationAxis = fillVectorFromLine(lines[16]);
	shape->rotationSpeed = (float)strtod(lines[17].c_str(),&end);
	shape->movementPoint = fillVectorFromLine(lines[18]);
	shape->movementAxis = fillVectorFromLine(lines[19]);
	shape->speed = (float)strtod(lines[20].c_str(),&end);
}

void loadBox(ifstream& in,Shape* shape)
{

}

void loadLight(ifstream& in, Light* light)
{
	string lines[6];

	for(int i=0;i<6;++i)
	{
		getline(in,lines[i]);
	}

	light->pos = fillVectorFromLine(lines[0]);
	light->specularColor = fillVectorFromLine(lines[1]);
	light->diffuseColor = fillVectorFromLine(lines[2]);
	light->ambientColor = fillVectorFromLine(lines[3]);
	Vector att = fillVectorFromLine(lines[4]);
	light->constantAttenuation = att.x;
	light->linearAttenuation = att.y;
	light->quadraticAttenuation = att.z;
	Vector i = fillVectorFromLine(lines[5]);
	light->ambient = i.x;
	light->specular = i.y;
	light->diffuse = i.z;
}

void loadScene(string file)
{
	cout << "Loading..." << file << endl;

	string line;
	ifstream in(file);

	numShapes = 0;
	numLights = 0;

	if(in.is_open())
	{
		char* end;
		getline(in, line);
		numShapes = strtol(line.c_str(),&end,10);
		getline(in, line);
		numLights = strtol(line.c_str(),&end,10);
		getline(in,line);

		for(int i=0;i<numShapes;++i)
		{
			getline(in,line);
			int s = strtol(line.c_str(),&end,10);
			if(s == sphere)
			{
				loadSphere(in,&shapes[i]);
			}
			else if(s == box)
			{
				loadBox(in,&shapes[i]);
			}
			getline(in,line);
		}
		for(int i=0;i<numLights;++i)
		{
			loadLight(in,&lights[i]);
		}
	}

}

void initShapes()
{
	numShapes = 102;
	numLights = 1;
	loadScene("scene1.txt");
	return;
/*
	shapes[0].type = sphere;
	shapes[0].pos.x = -3.0f;
	shapes[0].pos.y = 0.0f;
	shapes[0].pos.z = 0.0f;
	shapes[0].color.x = 1.0f;
	shapes[0].color.y = 1.0f;
	shapes[0].color.z = 1.0f;
	shapes[0].bounds.x = 3.0f;
	shapes[0].shininess = 20;
	shapes[0].reflectivity = .0f;
	shapes[0].texture.data = NULL;
	shapes[0].normalMap.data = NULL;
	shapes[0].bumpMap.data = NULL;
	shapes[0].reflectionMap.data = NULL;
	shapes[0].rotation.set(0.0f,0.0f,0.0f,1.0f);
	//shapes[0].rotation.rotateYawPitchRoll(180.0f,0.0f,0.0f);
	loadTexture("A:\\res\\earth.png", shapes[0].texture, CLAMP);
	//loadTexture("A:\\res\\earthnormal.png", shapes[0].normalMap, CLAMP);
	loadTexture("A:\\res\\earthspecular.png", shapes[0].specularMap, CLAMP);
		
		shapes[1].type = sphere;
		shapes[1].pos.x = 4.0f;
		shapes[1].pos.y = 0.0f;
		shapes[1].pos.z = 0.0f;
		shapes[1].color.x = 1.0f;
		shapes[1].color.y = 1.0f;
		shapes[1].color.z = 1.0f;
		shapes[1].bounds.x = 3.0f;
		shapes[1].shininess = 20;
		shapes[1].reflectivity = .0f;
		shapes[1].texture.data = NULL;
		shapes[1].normalMap.data = NULL;
		shapes[1].bumpMap.data = NULL;
		shapes[1].reflectionMap.data = NULL;
		shapes[1].rotation.set(0.0f,0.0f,0.0f,1.0f);
		shapes[1].rotation.rotateYawPitchRoll(180.0f,0.0f,0.0f);
		loadTexture("A:\\res\\earth.png", shapes[1].texture, CLAMP);
		//loadTexture("A:\\res\\earthnormal.png", shapes[1].normalMap, CLAMP);
		loadTexture("A:\\res\\earthspecular.png", shapes[1].specularMap, CLAMP);
	//numShapes = 107;
	numLights = 1;
	*/
//	int numSpheres = 102;

	float pos[] = { 
					///H
					-8.5f, 5.0f, 0.0f,
					-8.5f, 4.0f, 0.0f,
					-8.5f, 3.0f, 0.0f,
					-8.5f, 2.0f, 0.0f,
					-8.5f, 1.0f, 0.0f,

					-7.5f, 3.0f, 0.0f,

					-6.5f,5.0f, 0.0f,
					-6.5f,4.0f, 0.0f,
					-6.5f,3.0f, 0.0f,
					-6.5f,2.0f, 0.0f,
					-6.5f,1.0f, 0.0f,
					////E
					-5.0f, 5.0f, 0.0f,
					-5.0f, 4.0f, 0.0f,
					-5.0f, 3.0f, 0.0f,
					-5.0f, 2.0f, 0.0f,
					-5.0f, 1.0f, 0.0f,

					-4.0f, 5.0f, 0.0f,
					-4.0f, 3.0f, 0.0f,
					-4.0f, 1.0f, 0.0f,

					-3.0f, 5.0f, 0.0f,
					-3.0f, 1.0f, 0.0f,
					////L
					-1.5f, 5.0f, 0.0f,
					-1.5f, 4.0f, 0.0f,
					-1.5f, 3.0f, 0.0f,
					-1.5f, 2.0f, 0.0f,
					-1.5f, 1.0f, 0.0f,

					-0.5f, 1.0f, 0.0f,
					 0.5f, 1.0f, 0.0f,
					 ////L
					 2.0f, 5.0f, 0.0f,
					 2.0f, 4.0f, 0.0f,
					 2.0f, 3.0f, 0.0f,
					 2.0f, 2.0f, 0.0f,
					 2.0f, 1.0f, 0.0f,

					 3.0f, 1.0f, 0.0f,
					 4.0f, 1.0f, 0.0f,
					 ////O
					 5.5f, 5.0f, 0.0f,
					 5.5f, 4.0f, 0.0f,
					 5.5f, 3.0f, 0.0f,
					 5.5f, 2.0f, 0.0f,
					 5.5f, 1.0f, 0.0f,

					 6.5f, 5.0f, 0.0f,
					 6.5f, 1.0f, 0.0f,

					 7.5f, 5.0f, 0.0f,
					 7.5f, 4.0f, 0.0f,
					 7.5f, 3.0f, 0.0f,
					 7.5f, 2.0f, 0.0f,
					 7.5f, 1.0f, 0.0f,
					 ////W
					-9.5f, -5.0f, 0.0f,
					-9.5f, -4.0f, 0.0f,
					-9.5f, -3.0f, 0.0f,
					-9.5f, -2.0f, 0.0f,
					-9.5f, -1.0f, 0.0f,

					-8.5f,-5.0f, 0.0f,
					-7.5f,-4.0f, 0.0f,
					-7.5f,-3.0f, 0.0f,
					-6.5f,-5.0f, 0.0f,

					-5.5f,-5.0f, 0.0f,
					-5.5f,-4.0f, 0.0f,
					-5.5f,-3.0f, 0.0f,
					-5.5f,-2.0f, 0.0f,
					-5.5f,-1.0f, 0.0f,
					////O
					-4.0f,-5.0f, 0.0f,
					-4.0f,-4.0f, 0.0f,
					-4.0f,-3.0f, 0.0f,
					-4.0f,-2.0f, 0.0f,
					-4.0f,-1.0f, 0.0f,

					-3.0f,-5.0f, 0.0f,
					-3.0f,-1.0f, 0.0f,
					
					-2.0f,-5.0f, 0.0f,
					-2.0f,-4.0f, 0.0f,
					-2.0f,-3.0f, 0.0f,
					-2.0f,-2.0f, 0.0f,
					-2.0f,-1.0f, 0.0f,
					////R
					-0.5f,-5.0f, 0.0f,
					-0.5f,-4.0f, 0.0f,
					-0.5f,-3.0f, 0.0f,
					-0.5f,-2.0f, 0.0f,
					-0.5f,-1.0f, 0.0f,
					
					 0.5f,-1.0f, 0.0f,
					 0.5f,-3.0f, 0.0f,
					 0.5f,-4.0f, 0.0f,

					 1.5f,-1.0f, 0.0f,
					 1.5f,-2.0f, 0.0f,
					 1.5f,-3.0f, 0.0f,
					 1.5f,-5.0f, 0.0f,
					 ////L
					 3.0f,-5.0f, 0.0f,
					 3.0f,-4.0f, 0.0f,
					 3.0f,-3.0f, 0.0f,
					 3.0f,-2.0f, 0.0f,
					 3.0f,-1.0f, 0.0f,

					 4.0f,-5.0f, 0.0f,
					 5.0f,-5.0f, 0.0f,
					 ////D
					 6.5f,-5.0f, 0.0f,
					 6.5f,-4.0f, 0.0f,
					 6.5f,-3.0f, 0.0f,
					 6.5f,-2.0f, 0.0f,
					 6.5f,-1.0f, 0.0f,

					 7.5f,-5.0f, 0.0f,
					 7.5f,-1.0f, 0.0f,
					 
					 8.5f,-4.0f, 0.0f,
					 8.5f,-3.0f, 0.0f,
					 8.5f,-2.0f, 0.0f
				};

	for(int i=0;i<102;++i)
	{
		shapes[i].type = sphere;
		shapes[i].pos.x = pos[i*3] + 0.5f;
		shapes[i].pos.y = pos[i*3+1];
		shapes[i].pos.z = pos[i*3+2] - 4.0f;
		shapes[i].color.x = 1.0f;
		shapes[i].color.y = 1.0f;
		shapes[i].color.z = 1.0f;
		shapes[i].bounds.x = .5f;
		shapes[i].shininess = 30;
		shapes[i].reflectivity = 0.3f;
		shapes[i].texture.data = NULL;
		shapes[i].normalMap.data = NULL;
		shapes[i].bumpMap.data = NULL;
		shapes[i].reflectionMap.data = NULL;
		shapes[i].rotation.set(0.0f,0.0f,0.0f,1.0f);
		shapes[i].rotation.rotateYawPitchRoll(180.0f,0.0f,0.0f);
		loadTexture("g:\\old\\res\\earth.png", shapes[i].texture, CLAMP);
	}
	/*
	int num = numSpheres;

	shapes[num].type = box;
	shapes[num].pos.x = -10.0f;
	shapes[num].pos.y = 0.0f;
	shapes[num].pos.z = -5.0f;
	shapes[num].color.x = 0.6f;
	shapes[num].color.y = 0.0f;
	shapes[num].color.z = 0.0f;
	shapes[num].bounds.x = 0.000001f;
	shapes[num].bounds.y = 20.0f;
	shapes[num].bounds.z = 10.0f;
	shapes[num].shininess = 10;
	shapes[num].reflectivity = .2f;
	shapes[num].texture.data = NULL;
	shapes[num].normalMap.data = NULL;
	shapes[num].bumpMap.data = NULL;
	shapes[num].reflectionMap.data = NULL;

	num++;

	shapes[num].type = box;
	shapes[num].pos.x = 10.0f;
	shapes[num].pos.y = 0.0f;
	shapes[num].pos.z = -5.0f;
	shapes[num].color.x = 0.0f;
	shapes[num].color.y = 0.0f;
	shapes[num].color.z = 0.6f;
	shapes[num].bounds.x = 0.0001f;
	shapes[num].bounds.y = 20.0f;
	shapes[num].bounds.z = 10.0f;
	shapes[num].shininess = 10;
	shapes[num].reflectivity = .2f;
	shapes[num].texture.data = NULL;
	shapes[num].normalMap.data = NULL;
	shapes[num].bumpMap.data = NULL;
	shapes[num].reflectionMap.data = NULL;

	num++;

	shapes[num].type = box;
	shapes[num].pos.x = 0.0f;
	shapes[num].pos.y = -10.0f;
	shapes[num].pos.z = -5.0f;
	shapes[num].color.x = 0.6f;
	shapes[num].color.y = 0.6f;
	shapes[num].color.z = 0.6f;
	shapes[num].bounds.x = 20.0f;
	shapes[num].bounds.y = 0.00001f;
	shapes[num].bounds.z = 10.0f;
	shapes[num].shininess = 10;
	shapes[num].reflectivity = .2f;
	shapes[num].texture.data = NULL;
	shapes[num].normalMap.data = NULL;
	shapes[num].bumpMap.data = NULL;
	shapes[num].reflectionMap.data = NULL;

	num++;

	shapes[num].type = box;
	shapes[num].pos.x = 0.0f;
	shapes[num].pos.y = 10.0f;
	shapes[num].pos.z = -5.0f;
	shapes[num].color.x = 0.6f;
	shapes[num].color.y = 0.6f;
	shapes[num].color.z = 0.6f;
	shapes[num].bounds.x = 20.0f;
	shapes[num].bounds.y = 0.00001f;
	shapes[num].bounds.z = 10.0f;
	shapes[num].shininess = 10;
	shapes[num].reflectivity = .2f;
	shapes[num].texture.data = NULL;
	shapes[num].normalMap.data = NULL;
	shapes[num].bumpMap.data = NULL;
	shapes[num].reflectionMap.data = NULL;

	num++;

	shapes[num].type = box;
	shapes[num].pos.x = 0.0f;
	shapes[num].pos.y = 0.0f;
	shapes[num].pos.z = -15.0f;
	shapes[num].color.x = 0.6f;
	shapes[num].color.y = 0.6f;
	shapes[num].color.z = 0.6f;
	shapes[num].bounds.x = 10.0f;
	shapes[num].bounds.y = 10.0f;
	shapes[num].bounds.z = 0.00001f;
	shapes[num].shininess = 10;
	shapes[num].reflectivity = .2f;
	shapes[num].texture.data = NULL;
	shapes[num].normalMap.data = NULL;
	shapes[num].bumpMap.data = NULL;
	shapes[num].reflectionMap.data = NULL;
	*/
	
	lights[0].pos.x = 0.0f;
	lights[0].pos.y = 10.5f;
	lights[0].pos.z = 0.0f;

	lights[0].specularColor.x = .5f;
	lights[0].specularColor.y = .5f;
	lights[0].specularColor.z = .2f;

	lights[0].ambientColor.x = 1.0f;
	lights[0].ambientColor.y = 1.0f;
	lights[0].ambientColor.z = 1.0f;

	lights[0].diffuseColor.x = .5f;
	lights[0].diffuseColor.y = .5f;
	lights[0].diffuseColor.z = .5f;

	lights[0].ambient = 1.0f;
	lights[0].diffuse = .5f;
	lights[0].specular = .8f;
}

void initTracer()
{
	//recalcRays(camera);
}

int main(int argc, char **argv)
{
	srand((unsigned int)time(NULL));

    initGL(argc, argv);
	initCuda();
	initShapes();
	initTracer();
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	FEClock::init();

	feclock = new FEClock();
	feclock->update();

    glutMainLoop();
	atexit(cleanup);

    return 0;
}
