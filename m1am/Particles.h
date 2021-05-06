#pragma once

#include <vector>
#include "Vector3.h"

struct particle {
	unsigned int dyn_entity_id;
	struct vector3<float> velocity;
};

using namespace std;

extern vector<struct particle>		particles;

void particle_add(unsigned int particle_entity_id, struct vector3<float> position, struct vector3<float> orientation, float particle_speed);
void particles_tick();