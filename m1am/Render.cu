#include "Render.h"

#include "cuda_runtime.h"
#include "Matrix3.h"
#include "Vector3.h"
#include "Vector2.h"
#include "Camera.h"
#include "Entity.h"
#include "Grid.h"

#include "CUDAStreamHandler.h"

__global__ void render(	const unsigned int* bf_dynamic, const unsigned int cameras_position,				const unsigned int camera_c,		const unsigned int camera_id, const unsigned int entity_grid_position, const unsigned int entities_dynamic_position,
						const unsigned int* bf_static,	const unsigned int entities_static_position,		const unsigned int entities_static_count,
														const unsigned int triangles_static_position,		const unsigned int triangles_static_grid_position,
														const unsigned int textures_map_static_position,	const unsigned int textures_static_position) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	//if (i < total_size) {
		struct camera* cameras = (struct camera*)&bf_dynamic[cameras_position];
		/*
		int camera_id = 0;
		int pixel_from = 0;
		int pixel_to = 0;
		for (; camera_id < camera_c; camera_id++) {
			pixel_to += (cameras[camera_id].resolution[0] * cameras[camera_id].resolution[1]);
			if (i < pixel_to) break;
			pixel_from = pixel_to;
		}
		*/

		struct vector3<float> camera_position = cameras[camera_id].position;
		struct vector3<float> camera_orientation = cameras[camera_id].orientation;
		struct vector2<int> camera_resolution = cameras[camera_id].resolution;
		struct vector2<float> camera_fov = cameras[camera_id].fov;
		camera_fov[0] /= cameras[camera_id].digital_zoom;
		camera_fov[1] /= cameras[camera_id].digital_zoom;
		int image_x = i % (camera_resolution[0]/5);
		int image_y = i / (camera_resolution[0]/5);
		unsigned char* image_d = cameras[camera_id].device_ptr;
		unsigned int* image_d_ray = cameras[camera_id].device_ptr_ray;

		if (i < (camera_resolution[0] * camera_resolution[1])/5) {
			float closest = FLT_MAX;
			unsigned int closest_id_entity = UINT_MAX;
			unsigned int closest_id_triangle = UINT_MAX;
			unsigned int closest_id_texture_map = UINT_MAX;
			float closest_id_s = 0.0f;
			float closest_id_ts = 0.0f;

			float phi	= -camera_fov[0] / 2.0f + (image_x * (camera_fov[0] / (float)(camera_resolution[0]/5)));
			float theta = -camera_fov[1] / 2.0f + (image_y * (camera_fov[1] / (float)(camera_resolution[1]/5)));

			float theta_start = -camera_fov[1] / 2.0f;
			float theta_end = camera_fov[1] / 2.0f;

			struct vector3<float> camera_ray_orientation = {
				sinf(theta + camera_orientation[0])	*	-cosf(phi + camera_orientation[2]),
				sinf(theta + camera_orientation[0])	*	 sinf(phi + camera_orientation[2]),
			    cosf(theta + camera_orientation[0]),
			};

			struct vector3<float> camera_ray_grid_traversal_position = camera_position;
			int entity_static_grid_current_idx = grid_get_index(bf_dynamic, entity_grid_position, camera_ray_grid_traversal_position);
			while (entity_static_grid_current_idx != -1 && closest_id_entity == UINT_MAX) {
				struct grid* entity_static_grid = (struct grid*) &bf_dynamic[entity_grid_position + 1];
				unsigned int entities_static_iddata_position = bf_dynamic[entity_static_grid->data_position_in_bf + 1 + entity_static_grid_current_idx];
				if (entities_static_iddata_position > 0) {
					unsigned int entities_count = bf_dynamic[entities_static_iddata_position];

					for (int e = 0; e < entities_count; e++) {
						unsigned int entity_id = bf_dynamic[entities_static_iddata_position + 1 + e];

						struct entity* entities = nullptr;
						struct entity* entity_current = nullptr;

						if (entity_id == UINT_MAX) {
							break;
						} else if (entity_id < entities_static_count){
							entities = (struct entity*)&bf_static[entities_static_position];
							entity_current = &entities[entity_id];
						} else {
							entities = (struct entity*)&bf_dynamic[entities_dynamic_position];
							entity_current = &entities[entity_id - entities_static_count];
						}
						
						vector3<float> oc = camera_position - entity_current->position;
						float a = dot(camera_ray_orientation, camera_ray_orientation);
						float b = 2.0 * dot(oc, camera_ray_orientation);
						float c = dot(oc, oc) - (entity_current->radius * entity_current->radius);
						float discriminant = b * b - 4 * a * c;
						float lambda = 0;
						if (discriminant < 0) {
							continue;
						} else {
							lambda = (-b - sqrt(discriminant)) / (2.0 * a);
						}
						
						if (isnan(lambda)) {
							continue;
						}

						vector3<float> r_intersect = camera_position - -(camera_ray_orientation * lambda);

						struct vector3<float>* triangles_static = (struct vector3<float> *) &bf_static[triangles_static_position];
						struct vector3<float>* triangles		 = &triangles_static[entity_current->triangles_id];

						struct grid* triangle_static_grid = (struct grid*)&bf_static[entity_current->triangles_grid_id + 1];
						
						vector3<float> tg_grid_pos = r_intersect - entity_current->position;

						vector3<float> scale_rot = entity_current->scale;
						scale_rot = rotate_x(scale_rot, entity_current->orientation[0]);
						scale_rot = rotate_y(scale_rot, entity_current->orientation[1]);
						scale_rot = rotate_z(scale_rot, entity_current->orientation[2]);
						scale_rot = {
							abs(scale_rot[0]),
							abs(scale_rot[1]),
							abs(scale_rot[2]),
						};

						tg_grid_pos = { tg_grid_pos[0] / scale_rot[0], tg_grid_pos[1] / scale_rot[1], tg_grid_pos[2] / scale_rot[2] };
			
						vector3<float> tg_grid_dir = {
							camera_ray_orientation[0] / scale_rot[0],
							camera_ray_orientation[1] / scale_rot[1],
							camera_ray_orientation[2] / scale_rot[2]
						};
						
						tg_grid_pos = rotate_x(tg_grid_pos, entity_current->orientation[0]);
						tg_grid_dir = rotate_x(tg_grid_dir, entity_current->orientation[0]);

						tg_grid_pos = rotate_y(tg_grid_pos, entity_current->orientation[1]);
						tg_grid_dir = rotate_y(tg_grid_dir, entity_current->orientation[1]);

						tg_grid_pos = rotate_z(tg_grid_pos, entity_current->orientation[2]);
						tg_grid_dir = rotate_z(tg_grid_dir, entity_current->orientation[2]);
						
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
										} else {
											continue;
										}
									}
									r = a / b;
							
									if (r + lambda < 0.0 || r + lambda >= closest) {
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
										continue;
									}
									ts = (uv * wu - uu * wv) / D;
									if (ts < 0.0 || (s + ts) > 1.0) {
										continue;
									}
								
									r += lambda;
									
									closest = r;
									closest_id_entity = entity_id;
									closest_id_triangle = triangle_indices[t + 1];
									closest_id_s = s;
									closest_id_ts = ts;
									closest_id_texture_map = entity_current->texture_map_id;
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
				float grid_traverse_lambda = grid_traverse_in_direction(bf_dynamic, entity_grid_position, camera_ray_grid_traversal_position, camera_ray_orientation);
				int count = 0;
				do {
					camera_ray_grid_traversal_position = camera_ray_grid_traversal_position - (camera_ray_orientation * -(grid_traverse_lambda + (float)count * 1e-6f));
					entity_static_grid_new_idx = grid_get_index(bf_dynamic, entity_grid_position, camera_ray_grid_traversal_position);
					count++;
				} while (entity_static_grid_new_idx == entity_static_grid_current_idx);
				entity_static_grid_current_idx = entity_static_grid_new_idx;
			}
			if (closest < FLT_MAX) {
				image_d_ray[i * 2] = closest_id_entity;
				image_d_ray[i * 2 + 1] = closest_id_triangle;
			} else {
				image_d_ray[i * 2] = UINT_MAX;
				image_d_ray[i * 2 + 1] = UINT_MAX;
			}
		}
	//}
}

__global__ void render_conv(const unsigned int* bf_dynamic, const unsigned int cameras_position, const unsigned int camera_c, const unsigned int camera_id, const unsigned int entity_grid_position, const unsigned int entities_dynamic_position,
	const unsigned int* bf_static, const unsigned int entities_static_position, const unsigned int entities_static_count,
	const unsigned int triangles_static_position, const unsigned int triangles_static_grid_position,
	const unsigned int textures_map_static_position, const unsigned int textures_static_position) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	//if (i < total_size) {
	struct camera* cameras = (struct camera*)&bf_dynamic[cameras_position];
	/*
	int camera_id = 0;
	int pixel_from = 0;
	int pixel_to = 0;
	for (; camera_id < camera_c; camera_id++) {
		pixel_to += (cameras[camera_id].resolution[0] * cameras[camera_id].resolution[1]);
		if (i < pixel_to) break;
		pixel_from = pixel_to;
	}
	*/

	struct vector3<float> camera_position = cameras[camera_id].position;
	struct vector3<float> camera_orientation = cameras[camera_id].orientation;
	struct vector2<int> camera_resolution = cameras[camera_id].resolution;
	struct vector2<float> camera_fov = cameras[camera_id].fov;
	camera_fov[0] /= cameras[camera_id].digital_zoom;
	camera_fov[1] /= cameras[camera_id].digital_zoom;
	int image_x = i % (camera_resolution[0]);
	int image_y = i / (camera_resolution[0]);
	int image_src_x = image_x / 5;
	int image_src_y = image_y / 5;
	unsigned char* image_d = cameras[camera_id].device_ptr;
	unsigned int* image_d_ray = cameras[camera_id].device_ptr_ray;

	if (i < (camera_resolution[0] * camera_resolution[1])) {
		float closest = FLT_MAX;
		unsigned int closest_id_entity = UINT_MAX;
		unsigned int closest_id_triangle = UINT_MAX;
		unsigned int closest_id_texture_map = UINT_MAX;
		float closest_id_s = 0.0f;
		float closest_id_ts = 0.0f;

		float phi = -camera_fov[0] / 2.0f + (image_x * (camera_fov[0] / (float)camera_resolution[0]));
		float theta = -camera_fov[1] / 2.0f + (image_y * (camera_fov[1] / (float)camera_resolution[1]));

		struct vector3<float> camera_ray_orientation = {
			sinf(theta + camera_orientation[0]) * -cosf(phi + camera_orientation[2]),
			sinf(theta + camera_orientation[0]) * sinf(phi + camera_orientation[2]),
			cosf(theta + camera_orientation[0]),
		};

		bool entity_found = false;
		bool entity_hit = false;
		for (int x = -1; x < 1; x++) {
			for (int y = -1; y < 1; y++) {
				if (image_src_x + x >= 0 && image_src_x + x < camera_resolution[0] / 5 && image_src_y + y >= 0 && image_src_y + y < camera_resolution[1] / 5) {
					unsigned int entity_id = image_d_ray[(image_src_y +y) * (camera_resolution[0] / 5) * 2 + (image_src_x +x )* 2];
					unsigned int triangle_id = image_d_ray[(image_src_y + y)* (camera_resolution[0] / 5) * 2 + (image_src_x + x)* 2 + 1];
					if (entity_id < UINT_MAX) {
						entity_found = true;
						struct entity* entities = nullptr;
						struct entity* entity_current = nullptr;

						if (entity_id < entities_static_count) {
							entities = (struct entity*)&bf_static[entities_static_position];
							entity_current = &entities[entity_id];
						} else {
							entities = (struct entity*)&bf_dynamic[entities_dynamic_position];
							entity_current = &entities[entity_id - entities_static_count];
						}

						vector3<float> oc = camera_position - entity_current->position;
						float a = dot(camera_ray_orientation, camera_ray_orientation);
						float b = 2.0 * dot(oc, camera_ray_orientation);
						float c = dot(oc, oc) - (entity_current->radius * entity_current->radius);
						float discriminant = b * b - 4 * a * c;
						float lambda = 0;
						if (discriminant < 0) {
							continue;
						}
						else {
							lambda = (-b - sqrt(discriminant)) / (2.0 * a);
						}

						if (isnan(lambda)) {
							continue;
						}

						vector3<float> r_intersect = camera_position - -(camera_ray_orientation * lambda);

						struct vector3<float>* triangles_static = (struct vector3<float> *) & bf_static[triangles_static_position];
						struct vector3<float>* triangles = &triangles_static[entity_current->triangles_id];

						struct grid* triangle_static_grid = (struct grid*)&bf_static[entity_current->triangles_grid_id + 1];

						vector3<float> tg_grid_pos = r_intersect - entity_current->position;

						vector3<float> scale_rot = entity_current->scale;
						scale_rot = rotate_x(scale_rot, entity_current->orientation[0]);
						scale_rot = rotate_y(scale_rot, entity_current->orientation[1]);
						scale_rot = rotate_z(scale_rot, entity_current->orientation[2]);
						scale_rot = {
							abs(scale_rot[0]),
							abs(scale_rot[1]),
							abs(scale_rot[2]),
						};

						tg_grid_pos = { tg_grid_pos[0] / scale_rot[0], tg_grid_pos[1] / scale_rot[1], tg_grid_pos[2] / scale_rot[2] };

						vector3<float> tg_grid_dir = {
							camera_ray_orientation[0] / scale_rot[0],
							camera_ray_orientation[1] / scale_rot[1],
							camera_ray_orientation[2] / scale_rot[2]
						};

						tg_grid_pos = rotate_x(tg_grid_pos, entity_current->orientation[0]);
						tg_grid_dir = rotate_x(tg_grid_dir, entity_current->orientation[0]);

						tg_grid_pos = rotate_y(tg_grid_pos, entity_current->orientation[1]);
						tg_grid_dir = rotate_y(tg_grid_dir, entity_current->orientation[1]);

						tg_grid_pos = rotate_z(tg_grid_pos, entity_current->orientation[2]);
						tg_grid_dir = rotate_z(tg_grid_dir, entity_current->orientation[2]);

						vector3<float> u, v, n, dir, from, w0, w;
						float r;

						dir = tg_grid_dir;

						vector3<float> t0 = triangles[triangle_id * 3];
						vector3<float> t1 = triangles[triangle_id * 3 + 1];
						vector3<float> t2 = triangles[triangle_id * 3 + 2];

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

						if (r + lambda < 0.0 || r + lambda >= closest) {
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
							continue;
						}
						ts = (uv * wu - uu * wv) / D;
						if (ts < 0.0 || (s + ts) > 1.0) {
							continue;
						}

						entity_hit = true;
						r += lambda;

						closest = r;
						closest_id_entity = entity_id;
						closest_id_triangle = triangle_id;
						closest_id_s = s;
						closest_id_ts = ts;
						closest_id_texture_map = entity_current->texture_map_id;
					}
				}
			}
		}
		if (entity_found && !entity_hit) {
			for (int x = -2; x < 2; x++) {
				for (int y = -2; y < 2; y++) {
					if (x == 0 && y == 0) continue;
					if (image_src_x + x*2 >= 0 && image_src_x + x*2 < camera_resolution[0] / 5 && image_src_y + y*2 >= 0 && image_src_y + y*2 < camera_resolution[1] / 5) {
						unsigned int entity_id = image_d_ray[(image_src_y +y*2) * (camera_resolution[0] / 5) * 2 + (image_src_x + x*2)* 2];
						unsigned int triangle_id = image_d_ray[(image_src_y +y*2)* (camera_resolution[0] / 5) * 2 + (image_src_x + x*2)* 2 + 1];
						if (entity_id < UINT_MAX) {
							entity_found = true;

							struct entity* entities = nullptr;
							struct entity* entity_current = nullptr;

							if (entity_id < entities_static_count) {
								entities = (struct entity*)&bf_static[entities_static_position];
								entity_current = &entities[entity_id];
							} else {
								entities = (struct entity*)&bf_dynamic[entities_dynamic_position];
								entity_current = &entities[entity_id - entities_static_count];
							}

							vector3<float> oc = camera_position - entity_current->position;
							float a = dot(camera_ray_orientation, camera_ray_orientation);
							float b = 2.0 * dot(oc, camera_ray_orientation);
							float c = dot(oc, oc) - (entity_current->radius * entity_current->radius);
							float discriminant = b * b - 4 * a * c;
							float lambda = 0;
							if (discriminant < 0) {
								continue;
							}
							else {
								lambda = (-b - sqrt(discriminant)) / (2.0 * a);
							}

							if (isnan(lambda)) {
								continue;
							}

							vector3<float> r_intersect = camera_position - -(camera_ray_orientation * lambda);

							struct vector3<float>* triangles_static = (struct vector3<float> *) & bf_static[triangles_static_position];
							struct vector3<float>* triangles = &triangles_static[entity_current->triangles_id];

							struct grid* triangle_static_grid = (struct grid*)&bf_static[entity_current->triangles_grid_id + 1];

							vector3<float> tg_grid_pos = r_intersect - entity_current->position;

							vector3<float> scale_rot = entity_current->scale;
							scale_rot = rotate_x(scale_rot, entity_current->orientation[0]);
							scale_rot = rotate_y(scale_rot, entity_current->orientation[1]);
							scale_rot = rotate_z(scale_rot, entity_current->orientation[2]);
							scale_rot = {
								abs(scale_rot[0]),
								abs(scale_rot[1]),
								abs(scale_rot[2]),
							};

							tg_grid_pos = { tg_grid_pos[0] / scale_rot[0], tg_grid_pos[1] / scale_rot[1], tg_grid_pos[2] / scale_rot[2] };

							vector3<float> tg_grid_dir = {
								camera_ray_orientation[0] / scale_rot[0],
								camera_ray_orientation[1] / scale_rot[1],
								camera_ray_orientation[2] / scale_rot[2]
							};

							tg_grid_pos = rotate_x(tg_grid_pos, entity_current->orientation[0]);
							tg_grid_dir = rotate_x(tg_grid_dir, entity_current->orientation[0]);

							tg_grid_pos = rotate_y(tg_grid_pos, entity_current->orientation[1]);
							tg_grid_dir = rotate_y(tg_grid_dir, entity_current->orientation[1]);

							tg_grid_pos = rotate_z(tg_grid_pos, entity_current->orientation[2]);
							tg_grid_dir = rotate_z(tg_grid_dir, entity_current->orientation[2]);

							vector3<float> u, v, n, dir, from, w0, w;
							float r;

							dir = tg_grid_dir;

							vector3<float> t0 = triangles[triangle_id * 3];
							vector3<float> t1 = triangles[triangle_id * 3 + 1];
							vector3<float> t2 = triangles[triangle_id * 3 + 2];

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

							if (r + lambda < 0.0 || r + lambda >= closest) {
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
								continue;
							}
							ts = (uv * wu - uu * wv) / D;
							if (ts < 0.0 || (s + ts) > 1.0) {
								continue;
							}

							entity_hit = true;
							r += lambda;

							closest = r;
							closest_id_entity = entity_id;
							closest_id_triangle = triangle_id;
							closest_id_s = s;
							closest_id_ts = ts;
							closest_id_texture_map = entity_current->texture_map_id;
						}
					}
				}
			}
		}
		
		if (closest < FLT_MAX) {
			struct entity* entities = nullptr;
			struct entity* entity_current = nullptr;

			if (closest_id_entity < entities_static_count) {
				entities = (struct entity*)&bf_static[entities_static_position];
				entity_current = &entities[closest_id_entity];
			}
			else {
				entities = (struct entity*)&bf_dynamic[entities_dynamic_position];
				entity_current = &entities[closest_id_entity - entities_static_count];
			}

			struct triangle_texture* texture_map = (struct triangle_texture*)&bf_static[textures_map_static_position];
			struct triangle_texture* tt = &texture_map[entity_current->texture_map_id];

			unsigned char* textures = (unsigned char*)&bf_static[textures_static_position];
			unsigned char* texture_current = &textures[tt[closest_id_triangle].texture_id];

			unsigned int* texture_width = (unsigned int*)texture_current;
			unsigned int* texture_height = (unsigned int*)(texture_current + 4);

			struct vector2<float> uv_coord = {
				closest_id_s * *texture_width * tt[closest_id_triangle].texture_scale,
				(1 - closest_id_ts - closest_id_s) * *texture_height * tt[closest_id_triangle].texture_scale
			};

			float o_s = sinf(tt[closest_id_triangle].texture_orientation);
			float o_c = cosf(tt[closest_id_triangle].texture_orientation);

			struct vector2<float> uv_coord_rot = {
				uv_coord[0] * o_c - uv_coord[1] * o_s,
				uv_coord[0] * o_s + uv_coord[1] * o_c
			};

			int val_from_x = tt[closest_id_triangle].texture_coord[0] + (int)uv_coord_rot[0] % *texture_width;
			int val_from_y = tt[closest_id_triangle].texture_coord[1] + (int)uv_coord_rot[1] % *texture_height;

			int base_idx = 8 + val_from_y * *texture_width * 4 + val_from_x * 4;
			image_d[i * 4] = texture_current[base_idx + 0];
			image_d[i * 4 + 1] = texture_current[base_idx + 1];
			image_d[i * 4 + 2] = texture_current[base_idx + 2];
			image_d[i * 4 + 3] = 255;
		} else {
			image_d[i * 4] = (unsigned char)((camera_id+1) * 123);
			image_d[(i * 4) + 1] = (unsigned char)((camera_id + 1) * 77);
			image_d[(i * 4) + 2] = (unsigned char)((camera_id + 1) * 210);
			image_d[(i * 4) + 3] = 255;
		}
		
	}
	//}
}

void render_launch(		const unsigned int* bf_dynamic, const unsigned int cameras_position,				const unsigned int camera_c, const unsigned int camera_id, const unsigned int entity_grid_position, const unsigned int entities_dynamic_position,
						const unsigned int* bf_static,	const unsigned int entities_static_position,		const unsigned int entities_static_count,
														const unsigned int triangles_static_position,		const unsigned int triangles_static_grid_position,
														const unsigned int textures_map_static_position,	const unsigned int textures_static_position) {
	cudaError_t err = cudaSuccess;

	int threadsPerBlock = 512;
	int blocksPerGrid = ((cameras[camera_id].resolution[0] * cameras[camera_id].resolution[1])/5 + threadsPerBlock - 1) / threadsPerBlock;
	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	
	render <<<blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (
													bf_dynamic, cameras_position,				camera_c, camera_id, entity_grid_position, entities_dynamic_position,
													bf_static,	entities_static_position,		entities_static_count,
																triangles_static_position,		triangles_static_grid_position,
																textures_map_static_position,	textures_static_position);

	blocksPerGrid = (cameras[camera_id].resolution[0] * cameras[camera_id].resolution[1] + threadsPerBlock - 1) / threadsPerBlock;

	render_conv << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (
		bf_dynamic, cameras_position, camera_c, camera_id, entity_grid_position, entities_dynamic_position,
		bf_static, entities_static_position, entities_static_count,
		triangles_static_position, triangles_static_grid_position,
		textures_map_static_position, textures_static_position);

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

__global__ void compose_kernel(const unsigned char* src, const int src_width, const int src_height, const int src_channels, const int dx, const int dy, const int crop_x1, const int crop_x2, const int crop_y1, const int crop_y2, const int width, const int height, unsigned char* dst, const int dst_width, const int dst_height, const int dst_channels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height * dst_channels) {
		int base_row = (i / (dst_channels * width));
		int base_col = ((i % (dst_channels * width)) / dst_channels);

		int src_row = crop_y1 + base_row;
		int src_col = crop_x1 + base_col;

		int dst_row = dy + base_row;
		int dst_col = dx + base_col;
		int dst_channel = (i % (dst_channels * width)) % dst_channels;
		if (src_channels == dst_channels && dst_channels == 4) {
			if (dst_row >= 0 && dst_row < dst_height && dst_col >= 0 && dst_col < dst_width) {
				float alpha_value = src[src_row * src_width * src_channels + src_col * src_channels + 3];
				float value = src[src_row * src_width * src_channels + src_col * src_channels + dst_channel];
				if (dst_channel < 3) {
					float res_val = ((255.0f - (alpha_value)) / 255.0f) * dst[dst_row * (dst_width * dst_channels) + dst_col * dst_channels + dst_channel] + ((alpha_value) / 255.0f) * (value);
					dst[dst_row * (dst_width * dst_channels) + dst_col * dst_channels + dst_channel] = (unsigned char)res_val;
				} else {
					float cur_val = dst[dst_row * (dst_width * dst_channels) + dst_col * dst_channels + dst_channel];
					if (cur_val < value) dst[dst_row * (dst_width * dst_channels) + dst_col * dst_channels + dst_channel] = value;
				}
			}
		}
	}
}

void render_compose_kernel_launch(const unsigned char* src, const int src_width, const int src_height, const int src_channels, const int dx, const int dy, const int crop_x1, const int crop_x2, const int crop_y1, const int crop_y2, const int width, const int height, unsigned char* dst, const int dst_width, const int dst_height, const int dst_channels) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height * dst_channels + threadsPerBlock - 1) / threadsPerBlock;
	compose_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, src_width, src_height, src_channels, dx, dy, crop_x1, crop_x2, crop_y1, crop_y2, width, height, dst, dst_width, dst_height, dst_channels);
}

__constant__ float render_staircase_filter_r_k[9] = {
	 1.0f / 3.0f,	-1.0f / 4.0f,		 0.0f,
	-1.0f / 4.0f,	 1.0f / 3.0f,		-1.0f / 4.0f,
	 0.0f,			-1.0f / 4.0f,		 1.0f / 3.0f
};

__constant__ float render_staircase_filter_l_k[9] = {
	 0.0f,			-1.0f / 4.0f,		 1.0f / 3.0f,
	-1.0f / 4.0f,	 1.0f / 3.0f,		-1.0f / 4.0f,
	 1.0f / 3.0f,	-1.0f / 4.0f,		 0.0f
};

__global__ void render_staircase_filter_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const float amplify) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height) {
		int row = i / (width);
		int col = (i % (width));

		float value = 0.0f;

		for (int channel = 0; channel < channels; channel++) {
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					if (row + y >= 0 && row + y < height && col + x >= 0 && col + x < width) {
						float base_val = ((float)src[(row + y) * width * channels + (col + x) * channels + channel]);
						value += ((render_staircase_filter_r_k[(y + 1) * 3 + x + 1]) * base_val);
					}
				}
			}
		}
		value = value / 3.0f * amplify;
		if (value > 255.0f) value = 255.0f;

		dst[i] = (unsigned char)value;
	}
}

void render_staircase_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const float amplify) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	render_staircase_filter_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels, amplify);
	cudaStreamSynchronize(cuda_streams[3]);
}

__global__ void render_max_filter_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height) {
		int row = i / (width);
		int col = (i % (width));

		float value = 0.0f;

		for (int channel = 0; channel < channels; channel++) {
			for (int y = -k/2; y <= k/2; y++) {
				for (int x = -k/2; x <= k / 2; x++) {
					if (row + y >= 0 && row + y < height && col + x >= 0 && col + x < width) {
						float base_val = ((float)src[(row + y) * width * channels + (col + x) * channels + channel]);
						value = value * (base_val <= value) + base_val * (base_val > value);
					}
				}
			}
		}

		dst[i] = (unsigned char)value;
	}
}

void render_max_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int k) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	render_max_filter_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels, k);
	cudaStreamSynchronize(cuda_streams[3]);
}


__constant__ float render_edge_filter_k[9] = {
		-1 / 8.0f, -1 / 8.0f, -1 / 8.0f,
		-1 / 8.0f, 1		, -1 / 8.0f,
		-1 / 8.0f, -1 / 8.0f, -1 / 8.0f
};

__global__ void render_edge_filter_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const float amplify) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height) {
		int row = i / (width);
		int col = (i % (width));

		float value = 0.0f;

		for (int channel = 0; channel < channels; channel++) {
			for (int y = -1; y <= 1; y++) {
				for (int x = -1; x <= 1; x++) {
					if (row + y >= 0 && row + y < height && col + x >= 0 && col + x < width) {
						float base_val = ((float)src[(row + y) * width * channels + (col + x) * channels + channel]);
						value += ((render_edge_filter_k[(y + 1) * 3 + x + 1]) * base_val);
					}
				}
			}
		}
		value = value / 3.0f * amplify;
		if (value > 255.0f) value = 255.0f;

		dst[i] = (unsigned char)value;
	}
}

void render_edge_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const float amplify) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	render_edge_filter_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels, amplify);
	cudaStreamSynchronize(cuda_streams[3]);
}



__constant__ float render_gauss_blur_k[9] = {
		8.0f / 108.5f,	14.0f / 108.5f,	8.0f / 108.5f,
		14.0f / 108.5f,	20.5f / 108.5f,	14.0f / 108.5f,
		8.0f / 108.5f,	14.0f / 108.5f,	8.0f / 108.5f
};

__constant__ float render_gauss_blur_k5[25] = {
	1.0f/273.0f,	4.0f/273.0f,	7.0f / 273.0f,	4.0f/273.0f,	1.0f/273.0f,
	4.0f / 273.0f,	16.0f/273.0f,	26.0f / 273.0f,	16.0f / 273.0f,	4.0f / 273.0f,
	7.0f / 273.0f,	26.0f / 273.0f, 41.0f / 273.0f,	26.0f / 273.0f, 7.0f / 273.0f,
	4.0f / 273.0f,	16.0f / 273.0f,	26.0f / 273.0f,	16.0f / 273.0f,	4.0f / 273.0f,
	1.0f / 273.0f,	4.0f / 273.0f,	7.0f / 273.0f,	4.0f / 273.0f,	1.0f / 273.0f,
};

__global__ void render_gauss_blur_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height) {
		int row = i / (width);
		int col = (i % (width));

		for (int channel = 0; channel < channels; channel++) {
			float blurr_value = 0.0f;
			for (int y = -2; y <= 2; y++) {
				for (int x = -2; x <= 2; x++) {
					if (row + y >= 0 && row + y < height && col + x >= 0 && col + x < width) {
						float base_val = ((float)src[(row + y) * width * channels + (col + x) * channels + channel]);

						blurr_value += render_gauss_blur_k[(y + 2) * 3 + x + 2] * base_val;
					}
				}
			}
			if (blurr_value > 255.0f) blurr_value = 255.0f;
			dst[row * width * channels + col * channels + channel] = (unsigned char)blurr_value;
		}
	}
}

void render_gauss_blur_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	render_gauss_blur_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels);
	cudaStreamSynchronize(cuda_streams[3]);
}

__global__ void render_anti_alias_kernel(const unsigned char* src, const unsigned char *src_s, unsigned char* dst, const int width, const int height, const int channels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height) {
		int row = i / (width);
		int col = (i % (width));

		for (int channel = 0; channel < channels; channel++) {
			float blurr_value = 0.0f;
			for (int y = -2; y <= 2; y++) {
				for (int x = -2; x <= 2; x++) {
					if (row + y >= 0 && row + y < height && col + x >= 0 && col + x < width) {
						float base_val = ((float)src[(row + y) * width * channels + (col + x) * channels + channel]);

						blurr_value += render_gauss_blur_k[(y + 2) * 3 + x + 2] * base_val;
					}
				}
			}
			if (blurr_value > 255.0f) blurr_value = 255.0f;
			float dest_value = (255.0f - src_s[row * width + col]) / 255.0f * src[row * width * channels + col * channels + channel] + (src_s[row * width + col] / 255.0f)* blurr_value;
			dst[row * width * channels + col * channels + channel] = (unsigned char)dest_value;
		}
	}
}

void render_anti_alias_kernel_launch(const unsigned char* src, const unsigned char* src_s, unsigned char* dst, const int width, const int height, const int channels) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	render_anti_alias_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, src_s, dst, width, height, channels);
	cudaStreamSynchronize(cuda_streams[3]);
}