#pragma once

#include <string>
#include "BitField.h"

using namespace std;

struct level {
	struct bit_field	bf_static;

	unsigned int		entities_static_pos;
	unsigned int		entities_static_grid_pos;
	unsigned int		triangles_static_pos;
	unsigned int		triangles_static_grid_pos;
	unsigned int		textures_map_static_pos;
	unsigned int		textures_static_pos;

	//unsigned int		sin_position;
};

extern struct level* level_current;

void level_load(struct level *level, string name);
void level_save(struct level* level, string name);