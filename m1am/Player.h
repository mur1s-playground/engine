#pragma once

#include "Vector2.h"

struct player {
	unsigned int entity_id;
	unsigned int camera_id;
};

struct player_section {
	vector2<int>	d;
	vector2<int>	resolution;
};

extern struct player			*players;
extern unsigned int				player_squadsize;
extern unsigned int				player_squadcount;

extern unsigned char*			players_composed;

extern unsigned int				player_selected_id;

extern int						player_rotate_queue;

void players_init();
void players_render();