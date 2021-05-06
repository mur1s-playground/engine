#pragma once

#include "Vector3.h"

enum gun_type {
	GT_PISTOL,
	GT_AR,
	GT_SNIPER
};

struct gun {
	enum gun_type gt;

	unsigned int	fire_rate;
	unsigned int	fire_mode_active;
	bool			fire_released;
	unsigned int	last_shot;
	unsigned int	magazine_size;
	unsigned int	magazine_current;

	unsigned int	reload_time;
	unsigned int	reload_current;

	float			projectile_speed;

	unsigned int	particle_entity_id;
};

void gun_init(struct gun* g, enum gun_type gt);
void gun_tick(struct gun* g, struct vector3<float> position, struct vector3<float> orientation, bool shooting);
void gun_toggle_firemode(struct gun* g);