#ifndef ENTITY_HPP
#define ENTITY_HPP

#include <cuda_runtime.h>
#include "Vector3.hpp"

struct entity {
	struct vector3<float> 	position;
	struct vector3<float> 	orientation;
	struct vector3<float> 	scale;

	unsigned int 		triangles;
	float			radius;
	unsigned int		triangle_grid;

	unsigned int		texture_id;
};

void entity_init(struct entity *e, unsigned int triangles, float radius);

struct entity *entity_generate_cube(unsigned int triangles);

float *entity_generate_cube_triangles(unsigned int *out_len);

unsigned char *entity_generate_default_texture(int tw, int th, int t, int r, int g, int b);

#endif /* ENTITY_HPP */
