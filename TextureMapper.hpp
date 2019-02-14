#ifndef TEXTURE_MAPPER_HPP
#define TEXTURE_MAPPER_HPP

#include "BitField.hpp"

struct texture_mapper {
	struct bit_field textures;
};

void texture_mapper_init(struct texture_mapper *tm);

unsigned int texture_mapper_texture_add(struct texture_mapper *tm, unsigned char *texture, int tex_width, int tex_height);

void texture_mapper_register_device(struct texture_mapper *tm, unsigned int device_id);

void texture_mapper_update_device(struct texture_mapper *tm, unsigned int device_id);

#endif /* TEXTURE_MAPPER_HPP */
