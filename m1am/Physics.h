#pragma once

#include "Vector3.h"

struct physics {
	unsigned int				entity_id;

	struct vector3<float>		velocity;
	struct vector3<float>		acceleration;

	struct vector3<float>		position_next;

	unsigned int				hit_entity_id;
	unsigned int				hit_triangle_id;
};

extern unsigned int				physics_position_in_bf;
extern unsigned int				physics_size_in_bf;
extern struct physics*			physics_p;

void physics_init();
void physics_apply_force(unsigned int dyn_entity_id, struct vector3<float> acceleration);
void physics_invalidate_all();
void physics_tick(float dt);

void physics_launch(const float dt, unsigned int* bf_dynamic, const unsigned int entity_grid_position, const unsigned int entities_dynamic_position, const unsigned int physics_position, const unsigned int physics_count,
	const unsigned int* bf_static, const unsigned int entities_static_position, const unsigned int entities_static_count,
	const unsigned int triangles_static_position, const unsigned int triangles_static_grid_position);