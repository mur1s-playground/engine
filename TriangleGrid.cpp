#include "TriangleGrid.hpp"

#include "Vector2.hpp"
#include "Vector3.hpp"

#include <string.h>
#include <cassert>
#include "math.h"
#include <algorithm>
#include "stdlib.h"

//TODO: neighbour stuff must be worked on

unsigned int triangle_grid_init(struct bit_field *bf_triangles, int x_from, int x_to, float x_scale, int y_from, int y_to, float y_scale, int z_from, int z_to, float z_scale) {
	int x_size = (int)(x_to-x_from)/x_scale;
	assert(x_size > 0);
	int y_size = (int)(y_to-y_from)/y_scale;
	assert(y_size > 0);
	int z_size = (int)(z_to-z_from)/z_scale;
	assert(z_size > 0);

	unsigned int size = x_size*y_size*z_size;
	unsigned int position_in_bf = 0;

	position_in_bf = bit_field_add_bulk(bf_triangles, (unsigned int *) &size, 1);
	position_in_bf = bit_field_add_bulk_to_segment(bf_triangles, position_in_bf, (unsigned int *) &x_from, 1);
	position_in_bf = bit_field_add_bulk_to_segment(bf_triangles, position_in_bf, (unsigned int *) &x_to, 1);
	position_in_bf = bit_field_add_bulk_to_segment(bf_triangles, position_in_bf, (unsigned int *) &x_scale, ceil(sizeof(float)/(float)(sizeof(unsigned int))));
	position_in_bf = bit_field_add_bulk_to_segment(bf_triangles, position_in_bf, (unsigned int *) &y_from, 1);
        position_in_bf = bit_field_add_bulk_to_segment(bf_triangles, position_in_bf, (unsigned int *) &y_to, 1);
        position_in_bf = bit_field_add_bulk_to_segment(bf_triangles, position_in_bf, (unsigned int *) &y_scale, ceil(sizeof(float)/(float)(sizeof(unsigned int))));
	position_in_bf = bit_field_add_bulk_to_segment(bf_triangles, position_in_bf, (unsigned int *) &z_from, 1);
        position_in_bf = bit_field_add_bulk_to_segment(bf_triangles, position_in_bf, (unsigned int *) &z_to, 1);
        position_in_bf = bit_field_add_bulk_to_segment(bf_triangles, position_in_bf, (unsigned int *) &z_scale, ceil(sizeof(float)/(float)(sizeof(unsigned int))));

	unsigned int *tmp = (unsigned int *) malloc(size*sizeof(unsigned int));
        memset(tmp, 0, size*sizeof(unsigned int));

	position_in_bf = bit_field_add_bulk_to_segment(bf_triangles, position_in_bf, (unsigned int *) tmp, size);

	free(tmp);

	return position_in_bf;
}

void triangle_grid_add_triangle(struct bit_field *bf_triangles, unsigned int position_in_bf, const float *triangle, const unsigned int pos) {
	vector2<int> range_x = {(int) bf_triangles->data[position_in_bf+2], (int)bf_triangles->data[position_in_bf+3]};
	float range_x_scale = *((float *) &bf_triangles->data[position_in_bf+4]);
	vector2<int> range_y = {(int) bf_triangles->data[position_in_bf+5], (int)bf_triangles->data[position_in_bf+6]};
	float range_y_scale = *((float *) &bf_triangles->data[position_in_bf+7]);
	vector2<int> range_z = {(int) bf_triangles->data[position_in_bf+8], (int)bf_triangles->data[position_in_bf+9]};
	float range_z_scale = *((float *) &bf_triangles->data[position_in_bf+10]);

//	printf("ranges: %i %i %f %i %i %f %i %i %f\r\n", range_x[0], range_x[1], range_x_scale, range_y[0], range_y[1], range_y_scale, range_z[0], range_z[1], range_z_scale);

	vector3<float> v_1 = {triangle[0], triangle[1], triangle[2]};
	vector3<float> v_2 = {triangle[3], triangle[4], triangle[5]};
	vector3<float> v_3 = {triangle[6], triangle[7], triangle[8]};

	vector3<float> max_v = {std::max(v_1[0], std::max(v_2[0], v_3[0])), std::max(v_1[1], std::max(v_2[1], v_3[1])), std::max(v_1[2], std::max(v_2[2], v_3[2]))};
	vector3<float> min_v = {std::min(v_1[0], std::min(v_2[0], v_3[0])), std::min(v_1[1], std::min(v_2[1], v_3[1])), std::min(v_1[2], std::min(v_2[2], v_3[2]))};

	int x_neighbour_start = (min_v[0] - range_x[0])/range_x_scale;
	int y_neighbour_start = (min_v[1] - range_y[0])/range_y_scale;
	int z_neighbour_start = (min_v[2] - range_z[0])/range_z_scale;

	int x_neighbour_end = (max_v[0] - range_x[0])/range_x_scale;
	int y_neighbour_end = (max_v[1] - range_y[0])/range_y_scale;
	int z_neighbour_end = (max_v[2] - range_z[0])/range_z_scale;

//	printf("%i %i %i, %i %i %i\r\n", x_neighbour_start, y_neighbour_start, z_neighbour_start, x_neighbour_end, y_neighbour_end, z_neighbour_end);

	for (int i = x_neighbour_start; i <= x_neighbour_end; i++) {
		for (int j = y_neighbour_start; j <= y_neighbour_end; j++) {
			for (int k = z_neighbour_start; k <= z_neighbour_end; k++) {
				unsigned int cur_idx 	= (i)*((range_y[1]-range_y[0])*(range_z[1]-range_z[0])/(range_y_scale*range_z_scale))+((j)*(range_z[1]-range_z[0])/(range_z_scale))+k;
				unsigned int cur_val = bf_triangles->data[position_in_bf+11+cur_idx];
				if (cur_val == 0) {
					bit_field_update_data(bf_triangles, position_in_bf+11+cur_idx, bit_field_add_data(bf_triangles, pos));
				} else {
					unsigned int check_for_realloc = bit_field_add_data_to_segment(bf_triangles, cur_val, pos);
					if (check_for_realloc != cur_val) {
						bit_field_update_data(bf_triangles, position_in_bf+11+cur_idx, check_for_realloc);
					}
				}
			}
		}
	}
}
/*
void triangle_grid_remove_triangle(struct bit_field *bf, unsigned int position_in_bf, float *triangle, const unsigned int pos) {
        vector2<int> range_x = {(int) bf_triangles->data[position_in_bf+2], (int)bf_triangles->data[position_in_bf+3]};
        float range_x_scale = *((float *) &bf_triangles->data[position_in_bf+4]);
        vector2<int> range_y = {(int) bf_triangles->data[position_in_bf+5], (int)bf_triangles->data[position_in_bf+6]};
        float range_y_scale = *((float *) &bf_triangles->data[position_in_bf+7]);
        vector2<int> range_z = {(int) bf_triangles->data[position_in_bf+8], (int)bf_triangles->data[position_in_bf+9]};
        float range_z_scale = *((float *) &bf_triangles->data[position_in_bf+10]);

        vector3<float> v_1 = {triangle[0], triangle[1], triangle[2]};
        vector3<float> v_2 = {triangle[3], triangle[4], triangle[5]};
        vector3<float> v_3 = {triangle[6], triangle[7], triangle[8]};

        vector3<float> max = {std::max(v_1[0], std::max(v_2[0], v_3[0])), std::max(v_1[1], std::max(v_2[1], v_3[1])), std::max(v_1[2], std::max(v_2[2], v_3[2]))};
        vector3<float> min = {std::min(v_1[0], std::min(v_2[0], v_3[0])), std::min(v_1[1], std::min(v_2[1], v_3[1])), std::min(v_1[2], std::min(v_2[2], v_3[2]))};

        int x_neighbour_start = (min[0] - range_x[0])/range_x_scale;
        int y_neighbour_start = (min[1] - range_y[0])/range_y_scale;
        int z_neighbour_start = (min[2] - range_z[0])/range_z_scale;

        int x_neighbour_end = (max[0] - range_x[0])/range_x_scale;
        int y_neighbour_end = (max[1] - range_y[0])/range_y_scale;
        int z_neighbour_end = (max[2] - range_z[0])/range_z_scale;

        for (int i = x_neighbour_start; i <= x_neighbour_end; i++) {
                for (int j = y_neighbour_start; j <= y_neighbour_end; j++) {
                        for (int k = z_neighbour_start; k <= z_neighbour_end; k++) {
                                unsigned int cur_idx    = (i)*((range_y[1]-range_y[0])*(range_z[1]-range_z[0])/(range_y_scale*range_z_scale))+((j)*(range_z[1]-range_z[0])/(range_z_scale))+k;
                                unsigned int cur_val = bf_triangles->data[position_in_bf+11+cur_idx];
                                unsigned int check_for_realloc = bit_field_remove_data_from_segment(&bf_triangles, cur_val, pos);
				if (check_for_realloc == 0) {
					bit_field_update_data(&bf_triangles, position_in_bf+11+cur_idx, 0);
				}
                        }
                }
        }
}
*/
