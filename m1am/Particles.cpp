#include "Particles.h"
#include "Entity.h"
#include "M1am.h"
#include "Grid.h"
#include "Level.h"
#include "Physics.h"
#include <iostream>

vector<struct particle>		particles = vector<struct particle>();

void particle_add(unsigned int particle_entity_id, struct vector3<float> position, struct vector3<float> orientation, float particle_speed) {
	struct particle p;

	unsigned int dyn_entity_id = entity_dynamic_add(particle_entity_id, position, orientation, struct vector3<float>(1.0f, 1.0f, 1.0f));
	
	p.dyn_entity_id = dyn_entity_id;

	vector3<float> acceleration = {
		sinf(orientation[0]) * -cosf(orientation[2]),
		sinf(orientation[0]) * sinf(orientation[2]),
		cosf(orientation[0]),
	};

	acceleration = acceleration * particle_speed;

	physics_apply_force(dyn_entity_id, acceleration);

	particles.push_back(p);
}

void particles_tick() {
	entities_dynamic = (struct entity*)&bf_dynamic.data[bf_dynamic_m.entities_dynamic_position_in_bf];
	for (int p = particles.size() - 1; p >= 0; p--) {
		unsigned int dyn_entity_id = particles[p].dyn_entity_id;

		physics_p = (struct physics*)&bf_dynamic.data[physics_position_in_bf];

		struct vector3<float> position_next = physics_p[dyn_entity_id].position_next;
		
		if (physics_p[dyn_entity_id].hit_entity_id != UINT_MAX) {
			//cout << "hit" << std::endl;
		}

		if (length(position_next - entities_dynamic[dyn_entity_id].position) > 0) {
			if (grid_get_index(bf_dynamic.data, bf_dynamic_m.entity_grid_position_in_bf, position_next) == -1) {
				entity_dynamic_remove(dyn_entity_id, true);
				physics_p[dyn_entity_id].entity_id = UINT_MAX;
				physics_p[dyn_entity_id].velocity = { 0.0f, 0.0f, 0.0f };
				physics_p[dyn_entity_id].hit_entity_id = UINT_MAX;
				physics_p[dyn_entity_id].hit_triangle_id = UINT_MAX;
				particles.erase(particles.begin() + p);
			} else {
				entity_dynamic_move(dyn_entity_id, position_next);
				entities_dynamic = (struct entity*)&bf_dynamic.data[bf_dynamic_m.entities_dynamic_position_in_bf];
			}
		}
	}
	//improve
	bit_field_invalidate_bulk(&bf_dynamic, bf_dynamic_m.entities_dynamic_position_in_bf, (unsigned int)ceilf((bf_dynamic_m.entities_dynamic_allocated_count * sizeof(struct entity)) / (float)sizeof(unsigned int)));
}