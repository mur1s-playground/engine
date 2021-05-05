#include "Particles.h"
#include "Entity.h"
#include "M1am.h"
#include "Grid.h"
#include "Level.h"
#include <iostream>

vector<struct particle>		particles = vector<struct particle>();

void particle_add(struct vector3<float> position, struct vector3<float> orientation, struct vector3<float> velocity) {
	struct particle p;
	p.entity_id = entity_dynamic_add(8, position, orientation, struct vector3<float>(1.0f, 1.0f, 1.0f));
	p.velocity = {
		sinf(orientation[0]) * -cosf(orientation[2]),
		sinf(orientation[0]) * sinf(orientation[2]),
		cosf(orientation[0]),
	};
	p.velocity = p.velocity * 0.1f;
	particles.push_back(p);
}

void particles_tick() {
	entities_dynamic = (struct entity*)&bf_dynamic.data[bf_dynamic_m.entities_dynamic_position_in_bf];
	for (int p = particles.size() - 1; p >= 0; p--) {
		unsigned int dyn_entity_id = particles[p].entity_id;

		particles[p].velocity[2] -= 0.000091f;
		struct vector3<float> position_next = entities_dynamic[dyn_entity_id].position - -(particles[p].velocity);
		
		if (grid_get_index(bf_dynamic.data, bf_dynamic_m.entity_grid_position_in_bf, position_next) == -1) {
			entity_dynamic_remove(dyn_entity_id, true);
			particles.erase(particles.begin() + p);
		} else {
			entity_dynamic_move(dyn_entity_id, position_next);
			entities_dynamic = (struct entity*)&bf_dynamic.data[bf_dynamic_m.entities_dynamic_position_in_bf];
		}
	}
	//improve
	bit_field_invalidate_bulk(&bf_dynamic, bf_dynamic_m.entities_dynamic_position_in_bf, (unsigned int)ceilf((bf_dynamic_m.entities_dynamic_allocated_count * sizeof(struct entity)) / (float)sizeof(unsigned int)));
}