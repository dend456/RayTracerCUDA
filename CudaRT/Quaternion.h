#ifndef QUATERNION_H
#define QUATERNION_H

#include <cuda_runtime.h>
#include <math.h>
#include "Vector.h"

class Quaternion
{
	#define E 0.00001f

	#define PI 3.14159265358979f
	#define PIOVER180 0.01745329251f

	__device__ __host__ __inline__ float degreesToRadians(float a)
	{
		return a * PIOVER180;
	}
public:
	float x;
	float y;
	float z;
	float w;

	__device__ __host__ Quaternion(float ix, float iy, float iz, float iw) : x(ix),y(iy),z(iz),w(iw) {}
	__device__ __host__ Quaternion() {}

	__device__ __host__ void set(float ix, float iy, float iz, float iw)
	{
		x = ix;
		y = iy;
		z = iz;
		w = iw;
	}

	__device__ __host__ Vector mul(Vector& v)
	{
		Vector uv, uuv, qvec;
		Vector out(0.0f,0.0f,0.0f);

		qvec.x = x;
		qvec.y = y;
		qvec.z = z;

		v.cross(qvec,uv);
		uv.cross(qvec,uuv);

		uv.mul(2.0f * w);
		uuv.mul(2.0f);

		v.add(uv, out);
		out.add(uuv);
		return out;
	}

	__device__ __host__ void mul(float s)
	{
		x *= s;
		y *= s;
		z *= s;
		w *= s;
	}

	__device__ __host__ void mul(Quaternion& o)
	{
		x = w * o.x + x * o.w + y * o.z - z * o.y;
		y = w * o.y + y * o.w + z * o.x - x * o.z;
		z = w * o.z + z * o.w + x * o.y - y * o.x;
		w = w * o.w - x * o.x - y * o.y - z * o.z;
	}
	
	__device__ __host__ void rotateYawPitchRoll(float yaw, float pitch, float roll)
	{
		float	ex, ey, ez;		// temp half euler angles
		float	cr, cp, cy, sr, sp, sy, cpcy, spsy;		// temp vars in roll,pitch yaw

		ex = degreesToRadians(pitch) / 2.0f;	// convert to rads and half them
		ey = degreesToRadians(yaw) / 2.0f;
		ez = degreesToRadians(roll) / 2.0f;

		cr = cosf(ex);
		cp = cosf(ey);
		cy = cosf(ez);

		sr = sinf(ex);
		sp = sinf(ey);
		sy = sinf(ez);

		cpcy = cp * cy;
		spsy = sp * sy;

		w = cr * cpcy + sr * spsy;

		x = sr * cpcy - cr * spsy;
		y = cr * sp * cy + sr * cp * sy;
		z = cr * cp * sy - sr * sp * cy;

		normalize();
	}

	__device__ __host__ void rotateAxis(Vector& axis, float angle)
	{
		float rad = angle * 0.5f;
		float scale	= sinf(rad);

		w = cosf(rad);
		x = axis.x * scale;
		y = axis.y * scale;
		z = axis.z * scale;
	}

	__device__ __host__ float lengthSq()
	{
		return x * x + y * y + z * z + w * w;
	}

	__device__ __host__ float length()
	{
		return sqrtf(lengthSq());
	}

	__device__ __host__ void normalize()
	{
		float len = length();
		mul(1.0f / len);
	}

	__device__ __host__ Quaternion rotationBetweenVector(Vector& vec1, Vector& vec2)
	{
		Quaternion out(0.0f,0.0f,0.0f,1.0f);
		Vector v1(vec1);
		Vector v2(vec2);
		float a;

		v1.normalize();
		v2.normalize();
		
		a = v1.dot(v2);

		if (a >= 1.0) 
		{
			return out;
		}

		if (a < (1e-6f - 1.0f))	
		{			
			Vector axis;
			Vector X(1.0f,0.0f,0.0f);

			X.cross(axis, vec1);

			if (fabs(axis.length()) < E)
			{
				Vector Y(0.0f,1.0f,0.0f);
				Y.cross(axis, vec1);
			}

			axis.normalize();

			out.rotateAxis(axis, PI);
		} 
		else 
		{
			float s = sqrtf((1+a) * 2);
			float invs = 1 / s;

			Vector c;
			v1.cross(c, v2);

			out.x = c.x * invs;
			out.y = c.y * invs;
			out.z = c.z * invs;
			out.w = s * 0.5f;

			out.normalize();
		}

		return out;
	}

	__device__ __host__ Quaternion conjugate()
	{
		Quaternion out(-x,-y,-z,w);

		return out;
	}

	__device__ __host__ Quaternion inverse()
	{
		float l = length();
		Quaternion tmp, out(0.0f,0.0f,0.0f,0.0f);

		if (fabs(l) < E)
		{
			return out;
		}
		Quaternion conj = conjugate();

		conj.mul(1.0f / l);
		out = conj;
		return out;
	}
};

#endif