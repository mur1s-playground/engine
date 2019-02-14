#include <stdlib.h>

#include <stdio.h>
#include "World.hpp"
#include "math.h"

//TODO: something is wrong with entity grid scaling (add/remove)

void world_init(struct world *w) {
	w->cameras = 0;
	w->entities = 0;
	bit_field_init(&w->data, 2, 16);
	bit_field_init(&w->triangles, 1000, 128);
	entity_grid_init(&w->eg, -20, 20, 1.0, -20, 20, 1.0, -10, 10, 1.0);
//	entity_grid_init(&w->eg, 5, 16, 1.0, 46, 56, 1.0, -10, 10, 1.0);
//	entity_grid_init(&w->eg, 0, 32, 1.0, 26, 86, 1.0, -10, 10, 1.0);
	bit_field_init(&w->tgm.data, 2, 16);
	w->tgm.triangle_set_ids = 0;

	float sin_tmp = 0.0;
	w->sin_d = bit_field_add_bulk(&w->data, (unsigned int *) &sin_tmp, 1);
	for (int i = 1; i < 360; i++) {
		sin_tmp = sinf(i*M_PI/180.0);
		w->sin_d = bit_field_add_bulk_to_segment(&w->data, w->sin_d, (unsigned int *) &sin_tmp, 1);
	}
/*	for (int i = 0; i < 360; i++) {
		printf("%f ", *(float *)&w->data.data[w->sin_d+1+i]);
	}*/
}

unsigned int world_add_camera(struct world *w, struct camera *c) {
	if (w->cameras == 0) {
		w->cameras = bit_field_add_bulk(&w->data, (unsigned int *) c, ceil(sizeof(struct camera)/(float)sizeof(unsigned int)));
	} else {
		w->cameras = bit_field_add_bulk_to_segment(&w->data, w->cameras, (unsigned int *) c, ceil(sizeof(struct camera)/(float)sizeof(unsigned int)));
	}
	return w->data.data[w->cameras]/ceil(sizeof(struct camera)/(float)sizeof(unsigned int))-1;
}

void world_move_camera_to_position(struct world *w, unsigned int camera_id, const vector3<float> position) {
        struct camera *cameras = (struct camera *) &w->data.data[w->cameras+1];

	unsigned int offset = (int)((unsigned int *)&cameras[camera_id].position - (unsigned int *)&cameras[0])+1;
        if (cameras[camera_id].position[0] != position[0]) {
                bit_field_update_data(&w->data, w->cameras+offset, *(unsigned int *)&position[0]);
        }
        if (cameras[camera_id].position[1] != position[1]) {
                bit_field_update_data(&w->data, w->cameras+offset+1, *(unsigned int *)&position[1]);
        }
        if (cameras[camera_id].position[2] != position[2]) {
                bit_field_update_data(&w->data, w->cameras+offset+2, *(unsigned int *)&position[2]);
        }
}

void world_set_camera_orientation(struct world *w, unsigned int camera_id, const vector3<float> orientation) {
	struct camera *cameras = (struct camera *) &w->data.data[w->cameras+1];

	unsigned int offset = (unsigned int)((unsigned int *)&cameras[camera_id].orientation - (unsigned int *)&cameras[0])+1;
	if (cameras[camera_id].orientation[0] != orientation[0]) {
                bit_field_update_data(&w->data, w->cameras+offset, *(unsigned int *)&orientation[0]);
        }
        if (cameras[camera_id].orientation[1] != orientation[1]) {
                bit_field_update_data(&w->data, w->cameras+offset+1, *(unsigned int *)&orientation[1]);
        }
        if (cameras[camera_id].orientation[2] != orientation[2]) {
                bit_field_update_data(&w->data, w->cameras+offset+2, *(unsigned int *)&orientation[2]);
        }
}

unsigned int world_add_entity(struct world *w, struct entity *e) {
	// TriangleGrid
	// - refactor into other file
	// - maybe do offset calculation for vector2 everywhere
	int add = 0;
	unsigned int float_c = 0;
	unsigned int found_triangle_grid = 0;
	vector2<unsigned int> tr_grid_map = {e->triangles, 0};
	unsigned int v2_size_in_bf = (unsigned int) ceil(sizeof(vector2<unsigned int>)/(float)sizeof(unsigned int));
	if (w->tgm.triangle_set_ids == 0) {
		w->tgm.triangle_set_ids = bit_field_add_bulk(&w->tgm.data, (unsigned int *) &tr_grid_map, v2_size_in_bf);
		float_c = w->triangles.data[e->triangles];
		add = 1;
	} else {
		int found = 0;
		for (int i = 0; i < w->tgm.data.data[w->tgm.triangle_set_ids]/v2_size_in_bf; i++) {
			if (w->tgm.data.data[w->tgm.triangle_set_ids+1+(i*v2_size_in_bf)] == e->triangles) {
				found_triangle_grid = w->tgm.data.data[w->tgm.triangle_set_ids+1+1+(i*v2_size_in_bf)];
				found = 1;
				break;
			}
		}
		if (!found) {
			float_c = w->triangles.data[e->triangles];
			w->tgm.triangle_set_ids = bit_field_add_bulk_to_segment(&w->tgm.data, w->tgm.triangle_set_ids, (unsigned int *) &tr_grid_map, v2_size_in_bf);
			add = 1;
		}
	}
	if (add) {
		int triangles_c = float_c/9;
		int c_rad = ceil(e->radius);
		float scale = 1.0;
		e->triangle_grid = triangle_grid_init(&w->triangles, -c_rad, c_rad, scale, -c_rad, c_rad, scale, -c_rad, c_rad, scale);

//		printf("tr_grid init: %i, %f, %i\r\n", c_rad, scale, e->triangle_grid);
		unsigned int triangle_set_ids_size = w->tgm.data.data[w->tgm.triangle_set_ids]/v2_size_in_bf;
		bit_field_update_data(&w->tgm.data, w->tgm.triangle_set_ids+1+1+(triangle_set_ids_size-1)*(v2_size_in_bf), e->triangle_grid);

		float *floats = (float *)&w->triangles.data[e->triangles+1];
		for (int i = 0; i < triangles_c; i++) {
			triangle_grid_add_triangle(&w->triangles, e->triangle_grid, &floats[(i*9)], i*9);
		}
	} else {
		e->triangle_grid = found_triangle_grid;
	}

	if (w->entities == 0) {
                w->entities = bit_field_add_bulk(&w->data, (unsigned int *) e, ceil(sizeof(struct entity)/(float)sizeof(unsigned int)));
        } else {
                w->entities = bit_field_add_bulk_to_segment(&w->data, w->entities, (unsigned int *) e, ceil(sizeof(struct entity)/(float)sizeof(unsigned int)));
        }
        entity_grid_add_entity(&w->eg, e, w->data.data[w->entities]/ceil(sizeof(struct entity)/(float)sizeof(unsigned int))-1);

	return w->data.data[w->entities]/ceil(sizeof(struct entity)/(float)sizeof(unsigned int))-1;
}

//TODO: implement entity_grid_move, even if you don't want to
void world_move_entity_to_position(struct world *w, const unsigned int entity_id, const vector3<float> position) {
	struct entity *entities = (struct entity *) &w->data.data[w->entities+1];

        entity_grid_remove_entity(&w->eg, &entities[entity_id], entity_id);

	unsigned int offset = (unsigned int)((unsigned int *)&entities[entity_id].position - (unsigned int *)&entities[0])+1;
	if (entities[entity_id].position[0] != position[0]) {
		bit_field_update_data(&w->data, w->entities+offset, *(unsigned int *)&position[0]);
	}
	if (entities[entity_id].position[1] != position[1]) {
		bit_field_update_data(&w->data, w->entities+offset+1, *(unsigned int *)&position[1]);
	}
	if (entities[entity_id].position[2] != position[2]) {
                bit_field_update_data(&w->data, w->entities+offset+2, *(unsigned int *)&position[2]);
        }

	entity_grid_add_entity(&w->eg, &entities[entity_id], entity_id);
}

void world_set_entity_orientation(struct world *w, const unsigned int entity_id, const vector3<float> orientation) {
	struct entity *entities = (struct entity *) &w->data.data[w->entities+1];

	unsigned int offset = (unsigned int)((unsigned int *)&entities[entity_id].orientation - (unsigned int *)&entities[0])+1;
	if (entities[entity_id].orientation[0] != orientation[0]) {
                bit_field_update_data(&w->data, w->entities+offset, *(unsigned int *)&orientation[0]);
        }
        if (entities[entity_id].orientation[1] != orientation[1]) {
                bit_field_update_data(&w->data, w->entities+offset+1, *(unsigned int *)&orientation[1]);
        }
        if (entities[entity_id].orientation[2] != orientation[2]) {
                bit_field_update_data(&w->data, w->entities+offset+2, *(unsigned int *)&orientation[2]);
        }
}

void world_remove_entity(struct world *w, const unsigned int entity_id) {
	struct entity *entities = (struct entity *) &w->data.data[w->entities+1];

	entity_grid_remove_entity(&w->eg, &entities[entity_id], entity_id);
}

void world_set_texture_mapper(struct world *w, struct texture_mapper *tm) {
	w->tm = tm;
}

void world_register_device(struct world *w, unsigned int device_id) {
	bit_field_register_device(&w->data, device_id);
	bit_field_register_device(&w->triangles, device_id);
	entity_grid_register_device(&w->eg, device_id);
}

void world_update_device(struct world *w, unsigned int device_id) {
	bit_field_update_device(&w->data, device_id);
	entity_grid_update_device(&w->eg, device_id);
}
