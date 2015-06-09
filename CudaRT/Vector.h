#ifndef VECTOR_H
#define VECTOR_H

#include <cuda_runtime.h>
#include <math.h>

class Vector
{

public:
	float x;
	float y;
	float z;

	__device__ __host__ Vector(float ix, float iy, float iz) : x(ix),y(iy),z(iz) {}
	__device__ __host__ Vector(Vector& v) : x(v.x),y(v.y),z(v.z) {}
	__device__ __host__ Vector() {}

	__device__ __host__ __inline__ void sub(Vector& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
	}
	__device__ __host__ __inline__ void add(Vector& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
	}
	__device__ __host__ __inline__ void add(Vector& v, Vector& out)
	{
		out.x = x + v.x;
		out.y = y + v.y;
		out.z = z + v.z;
	}

	__device__ __host__ __inline__ void mul(Vector& v)
	{
		x *= v.x;
		y *= v.y;
		z *= v.z;
	}
	__device__ __host__ __inline__ void mul(float f)
	{
		x *= f;
		y *= f;
		z *= f;
	}
	__device__ __host__ __inline__ void div(float f)
	{
		x /= f;
		y /= f;
		z /= f;
	}

	__device__ __host__ __inline__ float dot(Vector& v)
	{
		return (x * v.x) + (y * v.y) + (z * v.z);
	}
	__device__ __host__ __inline__ void normalize()
	{
		float d = sqrt(x * x + y * y + z * z);
		if(d == 1){
			return;
		}
		d = 1 / d;
		x *= d;
		y *= d;
		z *= d;
	}

	__inline__ void normalizeHost()
	{
		float d = sqrt(x * x + y * y + z * z);
		if(d == 1){
			return;
		}
		d = 1 / d;
		x *= d;
		y *= d;
		z *= d;
	}

	__device__ __host__ void cross(Vector& o, Vector& out)
	{
		Vector v;

		v.x = (y * o.z) - (z * o.y);
		v.y = (z * o.x) - (x * o.z);
		v.z = (x * o.y) - (y * o.x);

		out.x = v.x;
		out.y = v.y;
		out.z = v.z;
	}

	__device__ __host__ __inline__ void copy(Vector* dst)
	{
		dst->x = x;
		dst->y = y;
		dst->z = z;
	}

	__device__ __host__ __inline__ float length()
	{
		return sqrt(x * x + y * y + z * z);
	}

};

#endif