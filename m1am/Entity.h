#pragma once

#include "Vector2.h"
#include "Vector3.h"

#include <string>

using namespace std;

struct triangle_texture {
	int						texture_id;

	vector2<int>			texture_coord;
	float					texture_orientation;
	float					texture_scale;
};

struct entity {
	struct vector3<float> 	position;
	struct vector3<float> 	orientation;
	struct vector3<float> 	scale;

	float					radius;

	unsigned int			triangles_c;

	unsigned int			triangles_id;
	unsigned int			triangles_grid_id;
	unsigned int			texture_map_id;
};