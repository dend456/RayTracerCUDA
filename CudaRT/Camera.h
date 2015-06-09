#ifndef CAMERA_H
#define CAMERA_H

class Camera
{
public:
	Vector pos;
	Vector lookat;
	Vector right;
	Vector down;

	Camera(Vector p, Vector la, Vector r, Vector d) : pos(p), lookat(la), right(r), down(d)
	{

	}

	Camera()
		: pos(Vector(0,0,20)), lookat(Vector(0,0,0)), right(Vector(1,0,0)), down(Vector(0,-1,0))
	{
	}
	
};

#endif