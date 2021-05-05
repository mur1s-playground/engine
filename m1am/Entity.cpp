#include "Entity.h"

#include "M1am.h"
#include "Level.h"
#include "Grid.h"

#include <iostream>

struct entity*			entities_dynamic = nullptr;
vector<unsigned int>	entities_dynamic_free = vector<unsigned int>();

void entity_dynamic_free_preallocate(unsigned int preallocate_count) {
	for (int i = bf_dynamic_m.entities_dynamic_allocated_count - 1; i >= 0; i--) {
		entities_dynamic_free.push_back(i);
	}
}

void entity_dynamic_free_clear() {
	entities_dynamic_free.clear();
}

void entity_dynamic_preallocate(unsigned int preallocate_count) {
	unsigned int size_in_bf = (unsigned int) ceilf(preallocate_count * sizeof(struct entity)/(float) sizeof(unsigned int));

	bf_dynamic_m.entities_dynamic_position_in_bf = bit_field_add_bulk_zero(&bf_dynamic, size_in_bf) + 1;
	bf_dynamic_m.entities_dynamic_allocated_count = preallocate_count;
	entity_dynamic_free_preallocate(preallocate_count);
}

unsigned int entity_dynamic_add(unsigned int static_entity_id, struct vector3<float> position, struct vector3<float> orientation, struct vector3<float> scale) {
	struct entity* entities = (struct entity *)&level_current->bf_static.data[level_current->entities_static_pos];
	struct entity* static_entity = &entities[static_entity_id];
	entities_dynamic = (struct entity *)&bf_dynamic.data[bf_dynamic_m.entities_dynamic_position_in_bf];

	bool space = false;
	int e = 0;
	int free_entities = entities_dynamic_free.size();
	if (free_entities > 0) {
		e = entities_dynamic_free[free_entities - 1];
		entities_dynamic_free.pop_back();
		space = true;
	}
	if (!space) {
		unsigned int size_to_add = bf_dynamic_m.entities_dynamic_allocated_count / 2;
		unsigned int size_in_bf = (unsigned int) ceilf(size_to_add * sizeof(struct entity) / (float)sizeof(unsigned int));
		unsigned int* tmp = (unsigned int*)malloc(size_to_add * sizeof(struct entity));
		memset(tmp, 0, size_to_add * sizeof(struct entity));
		bf_dynamic_m.entities_dynamic_position_in_bf = bit_field_add_bulk_to_segment(&bf_dynamic, bf_dynamic_m.entities_dynamic_position_in_bf - 1, tmp, size_in_bf, size_to_add * sizeof(struct entity)) + 1;
		free(tmp);
		entities_dynamic = (struct entity*)&bf_dynamic.data[bf_dynamic_m.entities_dynamic_position_in_bf];
		e = bf_dynamic_m.entities_dynamic_allocated_count;
		bf_dynamic_m.entities_dynamic_allocated_count += size_to_add;
		entities_dynamic_free.resize(free_entities + size_to_add);
		memcpy(entities_dynamic_free.data() + size_to_add, entities_dynamic_free.data(), size_to_add * sizeof(unsigned int));
		unsigned int* edf = entities_dynamic_free.data();
		for (int i = 0; i < size_to_add; i++) {
			*edf = bf_dynamic_m.entities_dynamic_allocated_count - 1 - i;
			edf++;
		}
	}

	memcpy(&entities_dynamic[e], static_entity, sizeof(struct entity));
	entities_dynamic[e].position = position;
	entities_dynamic[e].orientation = orientation;

	float max_s = entities_dynamic[e].scale[0];
	if (entities_dynamic[e].scale[1] > max_s) max_s = entities_dynamic[e].scale[1];
	if (entities_dynamic[e].scale[2] > max_s) max_s = entities_dynamic[e].scale[2];

	float radius_us = entities_dynamic[e].radius / max_s;
	struct vector3<float> e_radius = { radius_us, radius_us, radius_us };

	entities_dynamic[e].scale = scale;

	grid_object_add(&bf_dynamic, bf_dynamic.data, bf_dynamic_m.entity_grid_position_in_bf, entities_dynamic[e].position, entities_dynamic[e].scale, -e_radius, e_radius, level_current->entities_static_count + e);
	//improve
	bit_field_invalidate_bulk(&bf_dynamic, bf_dynamic_m.entities_dynamic_position_in_bf - 1, (unsigned int) ceilf((bf_dynamic_m.entities_dynamic_allocated_count * sizeof(struct entity))/(float) sizeof(unsigned int)));
	return e;
}

void entity_dynamic_remove(unsigned int entity_dynamic_id, bool from_grid) {
	entities_dynamic = (struct entity*)&bf_dynamic.data[bf_dynamic_m.entities_dynamic_position_in_bf];

	if (from_grid) {
		unsigned int e = entity_dynamic_id;

		float max_s = entities_dynamic[e].scale[0];
		if (entities_dynamic[e].scale[1] > max_s) max_s = entities_dynamic[e].scale[1];
		if (entities_dynamic[e].scale[2] > max_s) max_s = entities_dynamic[e].scale[2];

		float radius_us = entities_dynamic[e].radius / max_s;
		struct vector3<float> e_radius = { radius_us, radius_us, radius_us };

		grid_object_remove(&bf_dynamic, bf_dynamic.data, bf_dynamic_m.entity_grid_position_in_bf, entities_dynamic[e].position, entities_dynamic[e].scale, -e_radius, e_radius, level_current->entities_static_count + e);
	}

	entities_dynamic[entity_dynamic_id].radius = 0.0f;

	entities_dynamic_free.push_back(entity_dynamic_id);
}

void entity_dynamic_move(unsigned int entity_dynamic_id, struct vector3<float> position_to) {
	unsigned int e = entity_dynamic_id;

	float max_s = entities_dynamic[e].scale[0];
	if (entities_dynamic[e].scale[1] > max_s) max_s = entities_dynamic[e].scale[1];
	if (entities_dynamic[e].scale[2] > max_s) max_s = entities_dynamic[e].scale[2];

	float radius_us = entities_dynamic[e].radius / max_s;
	struct vector3<float> e_radius = { radius_us, radius_us, radius_us };

	grid_object_remove(&bf_dynamic, bf_dynamic.data, bf_dynamic_m.entity_grid_position_in_bf, entities_dynamic[entity_dynamic_id].position, entities_dynamic[entity_dynamic_id].scale, -e_radius, e_radius, level_current->entities_static_count + entity_dynamic_id);
	entities_dynamic[entity_dynamic_id].position = position_to;
	grid_object_add(&bf_dynamic, bf_dynamic.data, bf_dynamic_m.entity_grid_position_in_bf, entities_dynamic[entity_dynamic_id].position, entities_dynamic[entity_dynamic_id].scale, -e_radius, e_radius, level_current->entities_static_count + entity_dynamic_id);
}