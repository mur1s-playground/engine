#include "Player.h"

#include "M1am.h"
#include "Camera.h"
#include "Util.h"
#include "Render.h"
#include "Level.h"
#include "CUDAStreamHandler.h"
#include <iostream>
#include <algorithm>

struct player*							players					= nullptr;
unsigned int							player_squadsize		= 5;
unsigned int							player_squadcount		= 2;

unsigned int							player_selected_id		= 0;

unsigned int							players_size_in_bf		= (unsigned int)ceilf((player_squadcount * player_squadsize * sizeof(struct player))/(float) sizeof(unsigned int));
unsigned int							players_position_in_bf	= UINT_MAX;

vector<struct player_section>			players_section			= vector<struct player_section>();
vector<struct player_section>			players_section_		= vector<struct player_section>();
vector<int>								players_section_layer	= vector<int>();

unsigned char*							players_composed_device = nullptr;
unsigned char*							players_composed		= nullptr;

int										player_rotate_queue		= 0;

void players_init() {
	cudaMalloc(&players_composed_device, resolution[0] * resolution[1] * 4 * sizeof(unsigned char));
	players_composed = (unsigned char*) malloc(resolution[0] * resolution[1] * 4 * sizeof(unsigned char));

	float players_per_corner = (player_squadsize - 1) / 4.0f;

	struct player_section ps;

	ps.d = { 0, 0 };
	ps.resolution = resolution;
	players_section.push_back(ps);
	players_section_.push_back(ps);
	players_section_layer.push_back(0);

	ps.d = { resolution[0] - resolution[0] / 5, 0 };
	ps.resolution = resolution_section;
	players_section.push_back(ps);
	players_section_.push_back(ps);
	players_section_layer.push_back(1);
	
	ps.d = { resolution[0] - resolution[0] / 5, resolution[1] - resolution[1]/5 };
	players_section.push_back(ps);
	players_section_.push_back(ps);
	players_section_layer.push_back(2);

	ps.d = { 0, resolution[1] - resolution[1] / 5 };
	players_section.push_back(ps);
	players_section_.push_back(ps);
	players_section_layer.push_back(3);

	ps.d = { 0, 0 };
	players_section.push_back(ps);
	players_section_.push_back(ps);
	players_section_layer.push_back(4);

	cameras_size_in_bf = player_squadsize * sizeof(struct camera);
	cameras_position_in_bf = bit_field_add_bulk_zero(&bf_dynamic, cameras_size_in_bf) + 1;
	players_position_in_bf = bit_field_add_bulk_zero(&bf_dynamic, players_size_in_bf) + 1;

	players = (struct player *) &bf_dynamic.data[players_position_in_bf];
	cameras = (struct camera *) &bf_dynamic.data[cameras_position_in_bf];

	for (int sc = 0; sc < player_squadcount; sc++) {
		for (int ss = 0; ss < player_squadsize; ss++) {
			if (sc == 0) {
				struct camera *camera_player = &cameras[ss];

				camera_player->position = { 1.0f, -3.0f, 1.0f };
				camera_player->orientation = { M_PI / 2.0f, 0.0f, M_PI / 2.0f };
				camera_player->fov = { 90.0f * M_PI / 360.0f, 50.625f * M_PI / 360.0f };
				if (ss == player_selected_id) {
					camera_player->resolution = resolution;
				} else {
					camera_player->resolution = { resolution[0]/5, resolution[1]/5 };
				}
				camera_player->digital_zoom = 1.0f;
				cudaMalloc(&camera_player->device_ptr_ray, (resolution[0] * resolution[1])/5 * 2 * sizeof(unsigned int));
				cudaMalloc(&camera_player->device_ptr, resolution[0] * resolution[1] * 4 * sizeof(unsigned char));
			}
		}
	}
}

unsigned int players_rotation_tick_c		= 0;
unsigned int players_rotation_tick_total	= 30;

void players_rotation_tick() {
	for (int ss = 0; ss < player_squadsize; ss++) {
		unsigned int camera_id = (player_selected_id + ss) % player_squadsize;
		unsigned int camera_id_next = (player_selected_id + ss + 1) % player_squadsize;

		struct camera* c = &cameras[camera_id];
		players_section[camera_id].d = {
			(int)(players_section_[camera_id].d[0] + (players_section_[camera_id_next].d[0] - players_section_[camera_id].d[0]) * (players_rotation_tick_c / (float)players_rotation_tick_total)),
			(int)(players_section_[camera_id].d[1] + (players_section_[camera_id_next].d[1] - players_section_[camera_id].d[1]) * (players_rotation_tick_c / (float)players_rotation_tick_total)),
		};
		players_section[camera_id].resolution = {
			(int)(players_section_[camera_id].resolution[0] + (players_section_[camera_id_next].resolution[0] - players_section_[camera_id].resolution[0]) * ((players_rotation_tick_c / (float)players_rotation_tick_total))),
			(int)(players_section_[camera_id].resolution[1] + (players_section_[camera_id_next].resolution[1] - players_section_[camera_id].resolution[1]) * ((players_rotation_tick_c / (float)players_rotation_tick_total))),
		};
		c->resolution = players_section[camera_id].resolution;
	}
	if (players_rotation_tick_c == players_rotation_tick_total / 4) {
		rotate(players_section_layer.rbegin(), players_section_layer.rbegin() + 1, players_section_layer.rend());
	}
	bit_field_invalidate_bulk(&bf_dynamic, cameras_position_in_bf, cameras_size_in_bf);
	players_rotation_tick_c++;
	if (players_rotation_tick_c == players_rotation_tick_total + 1) {
		players_rotation_tick_c = 0;
		memcpy(players_section_.data(), players_section.data(), players_section.size() * sizeof(struct player_section));
		player_selected_id = (player_selected_id - 1 + player_squadsize) % player_squadsize;
		player_rotate_queue--;
	}
}

void players_render() {
	cudaMemsetAsync(players_composed_device, 0, resolution[0] * resolution[1] * 4 * sizeof(unsigned char), cuda_streams[3]);
	for (int ss = 0; ss < player_squadsize; ss++) {
		unsigned int camera_id = players_section_layer[ss];
		struct camera* c = &cameras[camera_id];

		render_launch(bf_dynamic.device_data[0], cameras_position_in_bf, player_squadsize, camera_id,
			level_current->bf_static.device_data[0], level_current->entities_static_pos, level_current->entities_static_grid_pos,
			level_current->triangles_static_pos, level_current->triangles_static_grid_pos,
			level_current->textures_map_static_pos, level_current->textures_static_pos);

		render_compose_kernel_launch(c->device_ptr, c->resolution[0], c->resolution[1], 4, players_section[camera_id].d[0], players_section[camera_id].d[1], 0, 0, 0, 0, c->resolution[0], c->resolution[1], players_composed_device, resolution[0], resolution[1], 4);
	}
	cudaStreamSynchronize(cuda_streams[3]);
	cudaMemcpy(players_composed, players_composed_device, resolution[0] * resolution[1] * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (player_rotate_queue) {
		players_rotation_tick();
	}
}