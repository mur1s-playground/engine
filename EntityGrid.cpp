#include "EntityGrid.hpp"

#include <string.h>
#include <cassert>

//TODO: neighbour stuff must be worked on

void entity_grid_init(struct entity_grid *eg, int x_from, int x_to, float x_scale, int y_from, int y_to, float y_scale, int z_from, int z_to, float z_scale) {
	int x_size = (int)(x_to-x_from)/x_scale;
	assert(x_size > 0);
	int y_size = (int)(y_to-y_from)/y_scale;
	assert(y_size > 0);
	int z_size = (int)(z_to-z_from)/z_scale;
	assert(z_size > 0);

	unsigned int size = x_size*y_size*z_size;

	bit_field_init(&eg->data, 2, 16);

	eg->grid = bit_field_add_bulk(&eg->data, (unsigned int *) &size, 1);
	eg->grid = bit_field_add_bulk_to_segment(&eg->data, eg->grid, (unsigned int *) &x_from, 1);
	eg->grid = bit_field_add_bulk_to_segment(&eg->data, eg->grid, (unsigned int *) &x_to, 1);
	eg->grid = bit_field_add_bulk_to_segment(&eg->data, eg->grid, (unsigned int *) &x_scale, ceil(sizeof(float)/(float)(sizeof(unsigned int))));
	eg->grid = bit_field_add_bulk_to_segment(&eg->data, eg->grid, (unsigned int *) &y_from, 1);
        eg->grid = bit_field_add_bulk_to_segment(&eg->data, eg->grid, (unsigned int *) &y_to, 1);
        eg->grid = bit_field_add_bulk_to_segment(&eg->data, eg->grid, (unsigned int *) &y_scale, ceil(sizeof(float)/(float)(sizeof(unsigned int))));
	eg->grid = bit_field_add_bulk_to_segment(&eg->data, eg->grid, (unsigned int *) &z_from, 1);
        eg->grid = bit_field_add_bulk_to_segment(&eg->data, eg->grid, (unsigned int *) &z_to, 1);
        eg->grid = bit_field_add_bulk_to_segment(&eg->data, eg->grid, (unsigned int *) &z_scale, ceil(sizeof(float)/(float)(sizeof(unsigned int))));

	unsigned int *tmp = (unsigned int *) malloc(size*sizeof(unsigned int));
        memset(tmp, 0, size*sizeof(unsigned int));

	eg->grid = bit_field_add_bulk_to_segment(&eg->data, eg->grid, (unsigned int *) tmp, size);

	free(tmp);
}

void entity_grid_add_entity(struct entity_grid *eg, const struct entity *e, const unsigned int pos) {
	vector2<int> range_x = {(int) eg->data.data[eg->grid+2], (int)eg->data.data[eg->grid+3]};
	float range_x_scale = *((float *) &eg->data.data[eg->grid+4]);
	vector2<int> range_y = {(int) eg->data.data[eg->grid+5], (int)eg->data.data[eg->grid+6]};
	float range_y_scale = *((float *) &eg->data.data[eg->grid+7]);
	vector2<int> range_z = {(int) eg->data.data[eg->grid+8], (int)eg->data.data[eg->grid+9]};
	float range_z_scale = *((float *) &eg->data.data[eg->grid+10]);
/*
	int eg_idx_x = (int)(floor(e->position[0]) - range_x[0])/range_x_scale;
        int eg_idx_y = (int)(floor(e->position[1]) - range_y[0])/range_y_scale;
        int eg_idx_z = (int)(floor(e->position[2]) - range_z[0])/range_z_scale;
*/

	int x_neighbour_start = ((e->position[0] - range_x[0] - (e->radius*e->scale[0])))/range_x_scale;
	int y_neighbour_start = ((e->position[1] - range_y[0] - (e->radius*e->scale[1])))/range_y_scale;
	int z_neighbour_start = ((e->position[2] - range_z[0] - (e->radius*e->scale[2])))/range_z_scale;

	int x_neighbour_end = ((e->position[0] - range_x[0] + (e->radius*e->scale[0])))/range_x_scale;
	int y_neighbour_end = ((e->position[1] - range_y[0] + (e->radius*e->scale[1])))/range_y_scale;
	int z_neighbour_end = ((e->position[2] - range_z[0] + (e->radius*e->scale[2])))/range_z_scale;

	for (int i = x_neighbour_start; i <= x_neighbour_end; i++) {
		for (int j = y_neighbour_start; j <= y_neighbour_end; j++) {
			for (int k = z_neighbour_start; k <= z_neighbour_end; k++) {
				unsigned int cur_idx 	= (i)*((range_y[1]-range_y[0])*(range_z[1]-range_z[0])/(range_y_scale*range_z_scale))+((j)*(range_z[1]-range_z[0])/(range_z_scale))+k;
				unsigned int cur_val = eg->data.data[eg->grid+11+cur_idx];
				if (cur_val == 0) {
					bit_field_update_data(&eg->data, eg->grid+11+cur_idx, bit_field_add_data(&eg->data, pos));
				} else {
					unsigned int check_for_realloc = bit_field_add_data_to_segment(&eg->data, cur_val, pos);
					if (check_for_realloc != cur_val) {
						bit_field_update_data(&eg->data, eg->grid+11+cur_idx, check_for_realloc);
					}
				}
			}
		}
	}
}

void entity_grid_remove_entity(struct entity_grid *eg, struct entity *e, const unsigned int pos) {
        vector2<int> range_x = {(int) eg->data.data[eg->grid+2], (int)eg->data.data[eg->grid+3]};
        float range_x_scale = *((float *) &eg->data.data[eg->grid+4]);
        vector2<int> range_y = {(int) eg->data.data[eg->grid+5], (int)eg->data.data[eg->grid+6]};
        float range_y_scale = *((float *) &eg->data.data[eg->grid+7]);
        vector2<int> range_z = {(int) eg->data.data[eg->grid+8], (int)eg->data.data[eg->grid+9]};
        float range_z_scale = *((float *) &eg->data.data[eg->grid+10]);

	int x_neighbour_start = ((e->position[0] - range_x[0] - (e->radius*e->scale[0])))/range_x_scale;
        int y_neighbour_start = ((e->position[1] - range_y[0] - (e->radius*e->scale[1])))/range_y_scale;
        int z_neighbour_start = ((e->position[2] - range_z[0] - (e->radius*e->scale[2])))/range_z_scale;

        int x_neighbour_end = ((e->position[0] - range_x[0] + (e->radius*e->scale[0])))/range_x_scale;
        int y_neighbour_end = ((e->position[1] - range_y[0] + (e->radius*e->scale[1])))/range_y_scale;
        int z_neighbour_end = ((e->position[2] - range_z[0] + (e->radius*e->scale[2])))/range_z_scale;

        for (int i = x_neighbour_start; i <= x_neighbour_end; i++) {
                for (int j = y_neighbour_start; j <= y_neighbour_end; j++) {
                        for (int k = z_neighbour_start; k <= z_neighbour_end; k++) {
                                unsigned int cur_idx    = (i)*((range_y[1]-range_y[0])*(range_z[1]-range_z[0])/(range_y_scale*range_z_scale))+((j)*(range_z[1]-range_z[0])/(range_z_scale))+k;
                                unsigned int cur_val = eg->data.data[eg->grid+11+cur_idx];
                                unsigned int check_for_realloc = bit_field_remove_data_from_segment(&eg->data, cur_val, pos);
				if (check_for_realloc == 0) {
					bit_field_update_data(&eg->data, eg->grid+11+cur_idx, 0);
				}
                        }
                }
        }
}

/*
int entity_grid_get_index_from_position(const struct entity_grid *eg, const vector3<float> position) {
	int key_x = (int)(floor(position[0]) - eg->range_x[0])/eg->range_x_scale;
	int key_y = (int)(floor(position[1]) - eg->range_y[0])/eg->range_y_scale;
	int key_z = (int)(floor(position[2]) - eg->range_z[0])/eg->range_z_scale;

	return (key_x*(eg->range_y[1]-eg->range_y[0])*(eg->range_z[1]-eg->range_z[0]))+(key_y*(eg->range_z[1]-eg->range_z[0]))+key_z;
}
*/

unsigned int entity_grid_register_device(struct entity_grid *eg, unsigned int device_id) {
	return bit_field_register_device(&eg->data, device_id);
}

void entity_grid_update_device(struct entity_grid *eg, unsigned int device_id) {
	bit_field_update_device(&eg->data, device_id);
}

void entity_grid_dump(const struct entity_grid *eg) {
/*	for (int i = 0; i < (eg->range_z[1]-eg->range_z[0])/eg->range_z_scale; i++) {
		printf("z: %f\r\n", eg->range_z[0]+i*eg->range_z_scale);
		for (int j = 0; j < (eg->range_y[1]-eg->range_y[0])/eg->range_y_scale; j++) {
			for (int k = 0; k < (eg->range_x[1]-eg->range_x[0])/eg->range_x_scale; k++) {
				int idx = (k*(eg->range_y[1]-eg->range_y[0])*(eg->range_z[1]-eg->range_z[0])/(eg->range_y_scale*eg->range_z_scale))+(j*(eg->range_z[1]-eg->range_z[0])/(eg->range_z_scale))+i;
				printf("%i\t", eg->grid[idx]);
			}
			printf("\r\n");
		}
	}*/
}














//LATER
struct entity *entity_grid_split_entity_by_grid(const struct entity_grid *eg, const struct entity *e, int *out_entities_c) { // for _very large_ stationary entities
	return NULL;
}
