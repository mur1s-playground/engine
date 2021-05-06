#include "M1am.h"

#include "Level.h"
#include "Camera.h"
#include "cuda_runtime.h"
#include "Util.h"
#include "Render.h"
#include "Entity.h"
#include "time.h"
#include "Player.h"
#include "Particles.h"
#include "Physics.h"
#include "Gun.h"

#include "SDLShow.h"
#include "lodepng.h"
#include <iostream>

struct bit_field					bf_dynamic;

struct vector2<int>					resolution			= { 1920, 1080 };
struct vector2<int>					resolution_section	= { 1920 / 5, 1080 / 5 };

struct bf_dynamic_meta				bf_dynamic_m;


int main(int argc, char* argv[]) {
	struct level l;

	level_save(&l, "default");
	level_load(&l, "default");

	players_init();
	physics_init();
	
	bit_field_update_device(&bf_dynamic, 0);


	sdl_show_window();
	SDL_Event sdl_event;

	int fps = 0;
	double sec = 0;

	struct vector2<int> mouse_position	= { resolution[0]/2, resolution[1]/2 };
	bool				capture_mouse	= false;
	bool				shift_pressed	= false;

	float				move_speed		= 1.0f;
	bool				move_forwards	= false;
	bool				move_back		= false;
	bool				move_left		= false;
	bool				move_right		= false;

	bool				shooting		= false;

	int						mouse_y_invert		= -1;
	struct vector2<float>	mouse_sensitivity	= { 1.0f, 1.0f };

	int						keybind_menu			= SDLK_ESCAPE;
	int						keybind_forward			= SDLK_UP;
	int						keybind_back			= SDLK_DOWN;
	int						keybind_left			= SDLK_LEFT;
	int						keybind_right			= SDLK_RIGHT;
	int						keybind_rotate			= SDLK_TAB;
	int						keybind_toggle_firemode = SDLK_HOME;

	float physics_time = 0.0f;

	while (true) {
		long tf = clock();
	
		bit_field_update_device(&bf_dynamic, 0);
		/*
		render_launch(	camera_player.resolution[0] * camera_player.resolution[1],
						camera_player_image_device[0],
						bf_dynamic.device_data[0], camera_player_pos, 1,
						l.bf_static.device_data[0], l.entities_static_pos, l.entities_static_grid_pos,
						l.triangles_static_pos, l.triangles_static_grid_pos,
						l.textures_map_static_pos, l.textures_static_pos);
		*/
		players_render();

		sdl_update_frame((void*)players_composed, capture_mouse);
		
		long tf_2 = clock();
		long tf_ = tf_2 - tf;
		double td = ((double)tf_ / CLOCKS_PER_SEC);
		sec += td;
		physics_time += td;
		fps++;

		if (sec >= 1.0) {
			printf("main fps: %d, main ft : % f\r\n", fps, tf_ / (double)CLOCKS_PER_SEC);
			sec = 0;
			fps = 0;
		}

		cameras = (struct camera*)&bf_dynamic.data[cameras_position_in_bf];

		while (SDL_PollEvent(&sdl_event) != 0) {
			if (sdl_event.type == SDL_MOUSEWHEEL) {
				bool up = sdl_event.wheel.y > 0;
				if (up) {
					players[player_selected_id].gun_active_id = 1;
				} else {
					players[player_selected_id].gun_active_id = 0;
				}
			}
			
			if (sdl_event.type == SDL_KEYDOWN) {
				if (sdl_event.key.keysym.sym == keybind_menu) {
					if (capture_mouse) {
						SDL_ShowCursor(SDL_ENABLE);
					}
					else {
						SDL_ShowCursor(SDL_DISABLE);
					}
					capture_mouse = !capture_mouse;
				} else if (sdl_event.key.keysym.sym == SDLK_LSHIFT || sdl_event.key.keysym.sym == SDLK_RSHIFT) {
					shift_pressed = true;
				} else if (sdl_event.key.keysym.sym == keybind_forward) {
					move_forwards = true;
				} else if (sdl_event.key.keysym.sym == keybind_back) {
					move_back = true;
				} else if (sdl_event.key.keysym.sym == keybind_left) {
					move_left = true;
				} else if (sdl_event.key.keysym.sym == keybind_right) {
					move_right = true;
				} else if (sdl_event.key.keysym.sym == keybind_rotate) {
					player_rotate_queue++;
				} else if (sdl_event.key.keysym.sym == keybind_toggle_firemode) {
					gun_toggle_firemode(&players[player_selected_id].gun[0]);
					cout << "toggling" << std::endl;
				}
			}

			if (sdl_event.type == SDL_KEYUP) {
				if (sdl_event.key.keysym.sym == SDLK_LSHIFT || sdl_event.key.keysym.sym == SDLK_RSHIFT) {
					shift_pressed = false;
				} else if (sdl_event.key.keysym.sym == keybind_forward) {
					move_forwards = false;
				} else if (sdl_event.key.keysym.sym == keybind_back) {
					move_back = false;
				} else if (sdl_event.key.keysym.sym == keybind_left) {
					move_left = false;
				} else if (sdl_event.key.keysym.sym == keybind_right) {
					move_right = false;
				}
			}

			if (capture_mouse) {
				cameras = (struct camera*)&bf_dynamic.data[cameras_position_in_bf];
				struct camera* p = &cameras[player_selected_id];

				if (sdl_event.type == SDL_MOUSEBUTTONDOWN && sdl_event.button.button == SDL_BUTTON(SDL_BUTTON_LEFT)) {
					shooting = true;
					cameras = (struct camera*)&bf_dynamic.data[cameras_position_in_bf];
				}

				if (sdl_event.type == SDL_MOUSEBUTTONUP && sdl_event.button.button == SDL_BUTTON(SDL_BUTTON_LEFT)) {
					shooting = false;
					cameras = (struct camera*)&bf_dynamic.data[cameras_position_in_bf];
				}
				p = &cameras[player_selected_id];

				if (sdl_event.type == SDL_MOUSEMOTION) {
					p->orientation = {
						p->orientation[0] + (sdl_event.motion.y - mouse_position[1]) * mouse_y_invert * mouse_sensitivity[1] * 1e-3f,
						p->orientation[1],
						p->orientation[2] + (sdl_event.motion.x - mouse_position[0]) * mouse_sensitivity[0] * 1e-3f,
					};
				}

				float o_s = sinf(-p->orientation[2] + M_PI / 2.0);
				float o_c = cosf(-p->orientation[2] + M_PI / 2.0);

				float dx = (-move_left + move_right) * move_speed * (float)td;
				float dy = (+move_forwards - move_back) * move_speed * (float)td;

				struct vector2<float> move_rot = {
					(dx * o_c - dy * o_s),
					(dx * o_s + dy * o_c)
				};

				p->position = {
						p->position[0] + move_rot[0],
						p->position[1] + move_rot[1],
						p->position[2]
				};
				bit_field_invalidate_bulk(&bf_dynamic, cameras_position_in_bf, cameras_size_in_bf);
			}
		
		}
		gun_tick(&players[player_selected_id].gun[players[player_selected_id].gun_active_id], cameras[player_selected_id].position, cameras[player_selected_id].orientation, shooting);
		physics_invalidate_all();

		bit_field_update_device(&bf_dynamic, 0);

		physics_tick(td);

		bit_field_update_host(&bf_dynamic, 0);
		particles_tick();
		
	}

	return 0;
}