#ifndef STRUCTS_H
#define STRUCTS_H

#include "Vector.h"
#include "Quaternion.h"

typedef struct Segment
{
	Vector pos;
	Vector dir;
} Segment;

typedef struct Image
{
	float* data;
	int width;
	int height;
	int wrapMode;
} Image;


typedef struct Shape
{
	unsigned int type;
	int shininess;
	float reflectivity;
	Vector pos;
	Vector bounds;
	Vector color;
	Image texture;
	Image normalMap;
	Image bumpMap;
	Image reflectionMap;
	Image specularMap;
	Quaternion rotation;
	Vector rotationAxis;
	float rotationSpeed;
	Vector movementPoint;
	Vector movementAxis;
	float speed;
} Shape;

typedef struct Light
{
	Vector pos;
	Vector specularColor;
	Vector diffuseColor;
	Vector ambientColor;

	float constantAttenuation;
	float linearAttenuation;
	float quadraticAttenuation;

	float ambient;
	float specular;
	float diffuse;
} Light;

#endif