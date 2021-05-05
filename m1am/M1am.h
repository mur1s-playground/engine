#pragma once

#include "BitField.h"
#include "Vector2.h"

struct bf_dynamic_meta {
	unsigned int entity_grid_position_in_bf;
	unsigned int entities_dynamic_position_in_bf;
	unsigned int entities_dynamic_allocated_count;
};

extern struct bit_field					bf_dynamic;

extern struct bf_dynamic_meta			bf_dynamic_m;


extern struct vector2<int>		resolution;
extern struct vector2<int>		resolution_section;