#pragma once

#include <vector>
#include "Vector3.h"

struct particle {
	unsigned int entity_id;
	struct vector3<float> velocity;
};

using namespace std;

extern vector<struct particle>		particles;

void particle_add(struct vector3<float> position, struct vector3<float> orientation, struct vector3<float> velocity);
void particles_tick();