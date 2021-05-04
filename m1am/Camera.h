#pragma once

#include "Vector2.h"
#include "Vector3.h"

struct camera {
	struct vector3<float>				position;
	struct vector3<float>				orientation;

	struct vector2<float>				fov;
	struct vector2<int>					resolution;

	float								digital_zoom;

	unsigned int*						device_ptr_ray;
	unsigned char*						device_ptr;
};

extern struct camera*					cameras;

extern unsigned int						cameras_size_in_bf;
extern unsigned int						cameras_position_in_bf;