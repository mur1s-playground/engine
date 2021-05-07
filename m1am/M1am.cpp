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
#include "Matrix3.h"
#include "Skeleton.h"

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

	unsigned test_id = entity_dynamic_add(0, struct vector3<float>(0.0f, 0.0f, 0.5f), struct vector3<float>(0.0f, 0.0f, 0.0f), 1.0f);

	unsigned long long tick = 0;

	while (true) {
		long tf = clock();
	
		bit_field_update_device(&bf_dynamic, 0);
		
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
				player_switch_gun(player_selected_id, up);
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
					gun_toggle_firemode(&players[player_selected_id].gun[players[player_selected_id].gun_active_id]);
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
				}

				if (sdl_event.type == SDL_MOUSEBUTTONUP && sdl_event.button.button == SDL_BUTTON(SDL_BUTTON_LEFT)) {
					shooting = false;
				}
				p = &cameras[player_selected_id];

				vector3<float> orientation_d = { 0.0f, 0.0f, 0.0f };

				if (sdl_event.type == SDL_MOUSEMOTION) {
					orientation_d = {
						(sdl_event.motion.y - mouse_position[1]) * mouse_y_invert * mouse_sensitivity[1] * 1e-3f,
						0.0f,
						(sdl_event.motion.x - mouse_position[0]) * mouse_sensitivity[0] * 1e-3f,
					};
				}

				vector3<float> position_d = {
					(-move_left + move_right)* move_speed * (float)td,
					(+move_forwards - move_back)* move_speed * (float)td,
					0.0f,
				};
				
				camera_move(player_selected_id, position_d, orientation_d);
				entity_dynamic_move(players[player_selected_id].entity_id - level_current->entities_static_count, p->position);
				entities_dynamic[players[player_selected_id].entity_id - level_current->entities_static_count].orientation = { 0.0f, 0.0f, p->orientation[2] };
			}
		}
		struct skeleton* sk = (struct skeleton*)&bf_dynamic.data[entities_dynamic[players[0].entity_id - level_current->entities_static_count].skeleton_id];
		skeleton_rotate_bone(sk, 0, struct vector2<float>(tick * 0.01f, 0.0f));
		
		entities_dynamic[test_id].orientation[2] += 0.01f;
		bit_field_invalidate_bulk(&bf_dynamic, bf_dynamic_m.entities_dynamic_position_in_bf, (bf_dynamic_m.entities_dynamic_allocated_count * sizeof(struct entity)) / (float) sizeof(unsigned int));
		//cout << cameras[player_selected_id].orientation[0] << " " << cameras[player_selected_id].orientation[1] << " " << cameras[player_selected_id].orientation[2] << " " << std::endl;
		//struct vector3<float> camera_to_entity_orientation = { 1.0f, 0.0f, 1.0f };
		gun_tick(&players[player_selected_id].gun[players[player_selected_id].gun_active_id], cameras[player_selected_id].position, cameras[player_selected_id].orientation, shooting);
		physics_invalidate_all();

		bit_field_update_device(&bf_dynamic, 0);

		physics_tick(td);

		bit_field_update_host(&bf_dynamic, 0);
		particles_tick();

		tick++;
	}

	return 0;
}