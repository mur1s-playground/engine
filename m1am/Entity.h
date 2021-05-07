#pragma once

#include "Vector2.h"
#include "Vector3.h"

#include <string>
#include <vector>

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
	float				 	scale;

	float					radius;

	unsigned int			triangles_c;

	unsigned int			triangles_id;
	unsigned int			triangles_grid_id;
	unsigned int			skeleton_id;
	unsigned int			texture_map_id;
};

extern struct entity*		entities_dynamic;

void entity_dynamic_free_preallocate(unsigned int preallocate_count);
void entity_dynamic_free_clear();
void entity_dynamic_preallocate(unsigned int preallocate_count);
unsigned int entity_dynamic_add(unsigned int static_entity_id, struct vector3<float> position, struct vector3<float> orientation, float scale);
void entity_dynamic_remove(unsigned int entity_dynamic_id, bool from_grid);
void entity_dynamic_move(unsigned int entity_dynamic_id, struct vector3<float> position_to);
