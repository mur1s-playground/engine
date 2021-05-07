#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"
#include "Physics.h"
#include "Entity.h"
#include "Grid.h"
#include "Matrix3.h"
#include "M1am.h"
#include "Level.h"
#include <iostream>

unsigned int				physics_position_in_bf = 0;
unsigned int				physics_size_in_bf = 0;
struct						physics* physics_p = nullptr;

void physics_init() {
	struct physics p;
	p.entity_id = UINT_MAX;
	p.velocity = { 0.0f, 0.0f, 0.0f };
	p.acceleration = { 0.0f, 0.0f, 0.0f };

	p.hit_entity_id = UINT_MAX;
	p.hit_triangle_id = UINT_MAX;

	physics_size_in_bf = (unsigned int)ceilf(bf_dynamic_m.entities_dynamic_allocated_count * sizeof(struct physics) / (float)sizeof(unsigned int));
	physics_position_in_bf = bit_field_add_bulk_zero(&bf_dynamic, physics_size_in_bf) + 1;
	physics_p = (struct physics *) &bf_dynamic.data[physics_position_in_bf];
	for (int i = 0; i < bf_dynamic_m.entities_dynamic_allocated_count; i++) {
		memcpy(&physics_p[i], &p, sizeof(struct physics));
	}
}

void physics_apply_force(unsigned int dyn_entity_id, struct vector3<float> acceleration) {
	physics_p = (struct physics*)&bf_dynamic.data[physics_position_in_bf];
	physics_p[dyn_entity_id].entity_id = level_current->entities_static_count + dyn_entity_id;
	physics_p[dyn_entity_id].acceleration = acceleration;
}

void physics_invalidate_all() {
	bit_field_invalidate_bulk(&bf_dynamic, physics_position_in_bf, physics_size_in_bf);
}

void physics_tick(float dt) {
	physics_launch(dt, bf_dynamic.device_data[0], bf_dynamic_m.entity_grid_position_in_bf, bf_dynamic_m.entities_dynamic_position_in_bf, physics_position_in_bf, bf_dynamic_m.entities_dynamic_allocated_count, level_current->bf_static.device_data[0], level_current->entities_static_pos, level_current->entities_static_count, level_current->triangles_static_pos, level_current->triangles_static_grid_pos);

	cudaStreamSynchronize(cuda_streams[3]);
}

__global__ void physics_collision(const float dt, unsigned int* bf_dynamic, const unsigned int entity_grid_position, const unsigned int entities_dynamic_position, const unsigned int physics_position, const unsigned int physics_count,
	const unsigned int* bf_static, const unsigned int entities_static_position, const unsigned int entities_static_count,
	const unsigned int triangles_static_position, const unsigned int triangles_static_grid_position) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < physics_count) {
		struct physics* physics_ = (struct physics *) &bf_dynamic[physics_position];
		struct physics* p = &physics_[i];
		
		struct entity* entities = nullptr;
		struct entity* entity_current = nullptr;

		if (p->entity_id == UINT_MAX) {
			return;
		} else if (p->entity_id < entities_static_count) {
			entities = (struct entity*)&bf_static[entities_static_position];
			entity_current = &entities[p->entity_id];
		} else {
			entities = (struct entity*)&bf_dynamic[entities_dynamic_position];
			entity_current = &entities[p->entity_id - entities_static_count];
		}

		struct entity* e_ = entity_current;

		p->acceleration[2] -= 0.000091f;
		p->velocity = p->velocity - -p->acceleration;
		p->acceleration = { 0.0f, 0.0f, 0.0f };

		float velocity_len = length(p->velocity);
		
		if (velocity_len > 0) {
			struct vector3<float> grid_traversal_position = e_->position;
			struct vector3<float> grid_traversal_direction = p->velocity/velocity_len;

			float closest = FLT_MAX;
			unsigned int closest_id_entity = UINT_MAX;
			unsigned int closest_id_triangle = UINT_MAX;

			int entity_static_grid_current_idx = grid_get_index(bf_dynamic, entity_grid_position, grid_traversal_position);
			while (entity_static_grid_current_idx != -1 && closest_id_entity == UINT_MAX) {
				struct grid* entity_static_grid = (struct grid*)&bf_dynamic[entity_grid_position + 1];
				unsigned int entities_static_iddata_position = bf_dynamic[entity_static_grid->data_position_in_bf + 1 + entity_static_grid_current_idx];
				if (entities_static_iddata_position > 0) {
					unsigned int entities_count = bf_dynamic[entities_static_iddata_position];

					for (int e = 0; e < entities_count; e++) {
						unsigned int entity_id = bf_dynamic[entities_static_iddata_position + 1 + e];
						if (entity_id == UINT_MAX) {
							break;
						} else if (entity_id < entities_static_count) {
							entities = (struct entity*)&bf_static[entities_static_position];
							entity_current = &entities[entity_id];
						} else {
							entities = (struct entity*)&bf_dynamic[entities_dynamic_position];
							entity_current = &entities[entity_id - entities_static_count];
						}

						vector3<float> oc = e_->position - entity_current->position;
						float a = dot(grid_traversal_direction, grid_traversal_direction);
						float b = 2.0 * dot(oc, grid_traversal_direction);
						float c = dot(oc, oc) - (entity_current->radius * entity_current->radius);
						float discriminant = b * b - 4 * a * c;
						float lambda = 0;
						if (discriminant < 0) {
							continue;
						} else {
							lambda = (-b - sqrt(discriminant)) / (2.0 * a);
						}

						if (isnan(lambda) || lambda > velocity_len) {
							continue;
						}

						vector3<float> r_intersect = e_->position - -(grid_traversal_direction * lambda);

						struct vector3<float>* triangles_static = (struct vector3<float> *) & bf_static[triangles_static_position];
						struct vector3<float>* triangles = &triangles_static[entity_current->triangles_id];

						struct grid* triangle_static_grid = (struct grid*)&bf_static[entity_current->triangles_grid_id + 1];

						vector3<float> tg_grid_pos = r_intersect - entity_current->position;

						tg_grid_pos = tg_grid_pos / entity_current->scale;

						vector3<float> tg_grid_dir = grid_traversal_direction / entity_current->scale;

						tg_grid_pos = rotate_x(tg_grid_pos, -entity_current->orientation[0]);
						tg_grid_dir = rotate_x(tg_grid_dir, -entity_current->orientation[0]);

						tg_grid_pos = rotate_y(tg_grid_pos, -entity_current->orientation[1]);
						tg_grid_dir = rotate_y(tg_grid_dir, -entity_current->orientation[1]);

						tg_grid_pos = rotate_z(tg_grid_pos, -entity_current->orientation[2]);
						tg_grid_dir = rotate_z(tg_grid_dir, -entity_current->orientation[2]);

						struct vector3<float> tg_grid_traversal_position = tg_grid_pos;
						int tg_grid_current_idx = grid_get_index(bf_static, entity_current->triangles_grid_id, tg_grid_pos);
						while (tg_grid_current_idx == -1) {
							tg_grid_pos = tg_grid_pos * 0.99f;
							tg_grid_current_idx = grid_get_index(bf_static, entity_current->triangles_grid_id, tg_grid_pos);
						}
						bool entity_hit = false;
						while (tg_grid_current_idx != -1) {
							unsigned int tg_current_val = bf_static[triangle_static_grid->data_position_in_bf + 1 + tg_grid_current_idx];
							if (tg_current_val > 0) {
								const unsigned int* triangle_indices = &bf_static[tg_current_val];
								unsigned int triangles_c = triangle_indices[0];
								for (int t = 0; t < triangles_c; t++) {
									vector3<float> u, v, n, dir, from, w0, w;
									float r, a, b;

									dir = tg_grid_dir;

									vector3<float> t0 = triangles[triangle_indices[t + 1] * 3];
									vector3<float> t1 = triangles[triangle_indices[t + 1] * 3 + 1];
									vector3<float> t2 = triangles[triangle_indices[t + 1] * 3 + 2];

									u = t1 - t0;
									v = t2 - t0;
									n = cross(u, v);

									from = tg_grid_pos;

									w0 = from - t0;

									a = -1 * dot(n, w0);
									b = dot(n, dir);
									if (fabs(b) < 0.00000001) {
										if (a == 0) {
											continue;
										}
										else {
											continue;
										}
									}
									r = a / b;

									//if (abs(r + lambda) < e_->radius) {

									//} else
									if (r + lambda < 0.0 || r + lambda >= closest) {
										//printf("r %f, lambda %f, %f %f %f\n", r, lambda, e_->position[0], e_->position[1], e_->position[2]);
										
										continue;
									}

									vector3<float> Il(from[0] + r * dir[0], from[1] + r * dir[1], from[2] + r * dir[2]);
									float uu, uv, vv, wu, wv, D;

									uu = dot(u, u);
									uv = dot(u, v);
									vv = dot(v, v);

									w = Il - t0;
									wu = dot(w, u);
									wv = dot(w, v);
									D = uv * uv - uu * vv;
									float s, ts;
									s = (uv * wv - vv * wu) / D;
									if (s < 0.0 || s > 1.0) {
										//printf("not in\n");
										continue;
									}
									ts = (uv * wu - uu * wv) / D;
									if (ts < 0.0 || (s + ts) > 1.0) {
										//printf("not in\n");
										continue;
									}

									r += lambda;

									//if (r - e_->radius > velocity_len) continue;
									//printf("hit");
									//printf("%f\n", r - velocity_len);

									closest = r;
									closest_id_entity = entity_id;
									closest_id_triangle = triangle_indices[t + 1];
									entity_hit = true;
								}
							}
							if (entity_hit) break;
							int tg_grid_new_idx = tg_grid_current_idx;
							float tg_grid_traverse_lambda = grid_traverse_in_direction(bf_static, entity_current->triangles_grid_id, tg_grid_traversal_position, tg_grid_dir);
							int count = 0;
							do {
								tg_grid_traversal_position = tg_grid_traversal_position - (tg_grid_dir * -(tg_grid_traverse_lambda + (float)count * 1e-6f));
								tg_grid_new_idx = grid_get_index(bf_static, entity_current->triangles_grid_id, tg_grid_traversal_position);
								count++;
							} while (tg_grid_new_idx == tg_grid_current_idx);
							tg_grid_current_idx = tg_grid_new_idx;
						}
					}
				}
				int entity_static_grid_new_idx = entity_static_grid_current_idx;
				float grid_traverse_lambda = grid_traverse_in_direction(bf_dynamic, entity_grid_position, grid_traversal_position, grid_traversal_direction);
				int count = 0;
				do {
					grid_traversal_position = grid_traversal_position - (grid_traversal_direction * -(grid_traverse_lambda + (float)count * 1e-6f));
					entity_static_grid_new_idx = grid_get_index(bf_dynamic, entity_grid_position, grid_traversal_position);
					count++;
				} while (entity_static_grid_new_idx == entity_static_grid_current_idx);
				entity_static_grid_current_idx = entity_static_grid_new_idx;
					
				//if (length(grid_traversal_position - e_->position) > velocity_len) break;
			}

			if (closest_id_entity != UINT_MAX) {
				p->position_next = e_->position - -(p->velocity / velocity_len * closest);
				p->hit_entity_id = closest_id_entity;
				p->hit_triangle_id = closest_id_triangle;
			} else {
				p->position_next = e_->position - -p->velocity;
			}
			p->velocity = p->velocity * 0.999999f;
		}
	}
}

void physics_launch(const float dt, unsigned int* bf_dynamic, const unsigned int entity_grid_position, const unsigned int entities_dynamic_position, const unsigned int physics_position, const unsigned int physics_count,
	const unsigned int* bf_static, const unsigned int entities_static_position, const unsigned int entities_static_count,
	const unsigned int triangles_static_position, const unsigned int triangles_static_grid_position) {
	cudaError_t err = cudaSuccess;

	int threadsPerBlock = 512;
	int blocksPerGrid = (physics_count + threadsPerBlock - 1) / threadsPerBlock;
	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	physics_collision << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (dt, bf_dynamic, entity_grid_position, entities_dynamic_position, physics_position, physics_count,
		bf_static, entities_static_position, entities_static_count,
		triangles_static_position, triangles_static_grid_position);

	/*
	cudaStreamSynchronize(cuda_streams[3]);

	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch render kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	*/
	//cudaDeviceSynchronize();

}