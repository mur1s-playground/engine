
#include <cuda_runtime.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Vector2.hpp"
#include "TextureMapper.hpp"

void texture_mapper_init(struct texture_mapper *tm) {
	bit_field_init(&tm->textures, 2, 1024);
}

unsigned int texture_mapper_texture_add(struct texture_mapper *tm, unsigned char *texture, int tex_width, int tex_height) {
	int cur_pos = 0;
	int tex_size = tex_width*tex_height*4*sizeof(unsigned char);
	vector2<int> tmp = {tex_width, tex_height};
	unsigned int pos = bit_field_add_bulk(&tm->textures, (unsigned int *) &tmp, 2);
	pos = bit_field_add_bulk_to_segment(&tm->textures, pos, (unsigned int *) texture, ceil(tex_size/(float)sizeof(unsigned int)));

	return pos;
}

void texture_mapper_register_device(struct texture_mapper *tm, unsigned int device_id) {
	bit_field_register_device(&tm->textures, device_id);
}

void texture_mapper_update_device(struct texture_mapper *tm, unsigned int device_id) {
	bit_field_update_device(&tm->textures, device_id);
}
