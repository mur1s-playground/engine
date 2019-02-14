#ifndef TRIANGLEGRID_HPP
#define TRIANGLEGRID_HPP

#include "BitField.hpp"

struct triangle_grid_meta {
	unsigned int triangle_set_ids;

	struct bit_field data;
};

unsigned int triangle_grid_init(struct bit_field *bf_triangles, int x_from, int x_to, float x_scale, int y_from, int y_to, float y_scale, int z_from, int z_to, float z_scale);
void triangle_grid_add_triangle(struct bit_field *bf_triangles, unsigned int position_in_bf, const float *triangle, const unsigned int pos);

#endif /* TRIANGLEGRID_HPP */
