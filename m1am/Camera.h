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
	
	struct vector3<float>				camera_ray_orientation_lt;
	struct vector3<float>				camera_ray_orientation_dt;
	struct vector3<float>				camera_ray_orientation_lb;
	struct vector3<float>				camera_ray_orientation_db;
};

extern struct camera*					cameras;

extern unsigned int						cameras_c;
extern unsigned int						cameras_size_in_bf;
extern unsigned int						cameras_position_in_bf;

void camera_move(unsigned int camera_id, vector3<float> position_d, vector3<float> orientation_d);