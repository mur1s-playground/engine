#include "Gun.h"
#include "Camera.h"
#include "Particles.h"
#include <iostream>

void gun_init(struct gun* g, enum gun_type gt) {
	if (gt == GT_PISTOL) {
		g->gt					= GT_PISTOL;
		g->particle_entity_id	= 0;

		g->fire_rate			= 20;
		g->magazine_size		= 7;
		g->reload_time			= 66;
		g->projectile_speed		= 0.1f;
		g->fire_mode_active		= 0;
	} else if (gt == GT_AR) {
		g->gt					= GT_AR;
		g->particle_entity_id	= 1;

		g->fire_rate			= 15;
		g->magazine_size		= 20;
		g->reload_time			= 100;
		g->projectile_speed		= 0.2f;
		g->fire_mode_active		= 1;
	} else if (gt == GT_SNIPER) {
		g->gt					= GT_SNIPER;
		g->particle_entity_id	= 2;

		g->fire_rate			= 50;
		g->magazine_size		= 10;
		g->reload_time			= 120;
		g->projectile_speed		= 0.5f;
		g->fire_mode_active		= 0;
	}
	g->fire_released = true;
	g->magazine_current = g->magazine_size;
	g->reload_current = 0;
	g->last_shot = g->fire_rate;
}

void gun_tick(struct gun* g, struct vector3<float> position, struct vector3<float> orientation, bool shooting) {
	if (shooting && g->last_shot == g->fire_rate && g->magazine_current > 0 && g->reload_current == 0 && (g->fire_mode_active == 1 || (g->fire_mode_active == 0 && g->fire_released))) {
		particle_add(g->particle_entity_id, position, orientation, g->projectile_speed);
		g->last_shot = 0;
		g->magazine_current--;
		g->fire_released = false;
	} else {
		if (!shooting) g->fire_released = true;
		if (g->last_shot < g->fire_rate) g->last_shot++;
		if (g->reload_current > 0) g->reload_current--;
		if (g->reload_current == 1) {
			g->magazine_current = g->magazine_size;
		}
		if (g->magazine_current == 0 && g->reload_current == 0) {
			g->reload_current = g->reload_time;
		}
	}
}

void gun_toggle_firemode(struct gun* g) {
	if (g->gt == GT_AR) {
		if (g->fire_mode_active) {
			g->fire_mode_active = 0;
		} else {
			g->fire_mode_active = 1;
		}
	}
}