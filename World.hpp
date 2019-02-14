#ifndef WORLD_HPP
#define WORLD_HPP

#include "TextureMapper.hpp"
#include "Camera.hpp"
#include "Entity.hpp"
#include "BitField.hpp"
#include "EntityGrid.hpp"
#include "TriangleGrid.hpp"
#include "Vector3.hpp"

struct world {
	unsigned int 			cameras;
	unsigned int 			entities;

	struct bit_field 		data;

	struct bit_field		triangles;

	struct texture_mapper		*tm;

	struct entity_grid 		eg;
	struct triangle_grid_meta	tgm;

	unsigned int			sin_d;
};

void world_init(struct world *w);

unsigned int world_add_camera(struct world *w, struct camera *c);
void world_move_camera_to_position(struct world *w, unsigned int camera_id, const vector3<float> position);
void world_set_camera_orientation(struct world *w, unsigned int camera_id, const vector3<float> orientation);

unsigned int world_add_entity(struct world *w, struct entity *e);
void world_move_entity_to_position(struct world *w, const unsigned int entity_id, const vector3<float> position);
void world_set_entity_orientation(struct world *w, const unsigned int entity_id, const vector3<float> orientation);
void world_remove_entity(struct world *w, const unsigned int entity_id);

void world_set_texture_mapper(struct world *w, struct texture_mapper *tm);

void world_register_device(struct world *w, unsigned int device_id);
void world_update_device(struct world *w, unsigned int device_id);

#endif /* WORLD_HPP */
