#ifndef ENTITYGRID_HPP
#define ENTITYGRID_HPP

#include "Vector2.hpp"
#include "Vector3.hpp"
#include "Entity.hpp"
#include "BitField.hpp"

struct entity_grid {
	unsigned int grid;

	struct bit_field 	data;
};

void entity_grid_init(struct entity_grid *eg, int x_from, int x_to, float x_scale, int y_from, int y_to, float y_scale, int z_from, int z_to, float z_scale);

void entity_grid_add_entity(struct entity_grid *eg, const struct entity *e, const unsigned int pos);
void entity_grid_remove_entity(struct entity_grid *eg, struct entity *e, const unsigned int pos);

//int entity_grid_get_index_from_position(const struct entity_grid *eg, const vector3<float> position);

unsigned int entity_grid_register_device(struct entity_grid *eg, unsigned int device_id);
void entity_grid_update_device(struct entity_grid *eg, unsigned int device_id);

void entity_grid_dump(const struct entity_grid *eg);

#endif /* ENTITYGRID_HPP */
