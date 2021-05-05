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


	int						mouse_y_invert		= -1;
	struct vector2<float>	mouse_sensitivity	= { 1.0f, 1.0f };

	int						keybind_menu		= SDLK_ESCAPE;
	int						keybind_forward		= SDLK_UP;
	int						keybind_back		= SDLK_DOWN;
	int						keybind_left		= SDLK_LEFT;
	int						keybind_right		= SDLK_RIGHT;
	int						keybind_rotate		= SDLK_TAB;
	
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

		//render_staircase_filter_kernel_launch(camera_player_image_device[0], camera_player_edge_filter_device[0], camera_player.resolution[0], camera_player.resolution[1], 3, 1.0f);

		//render_edge_filter_kernel_launch(camera_player_image_device[0], camera_player_edge_filter_device[0], camera_player.resolution[0], camera_player.resolution[1], 3, 3.0f);
		//render_max_filter_kernel_launch(camera_player_edge_filter_device[0], camera_player_edge_filter_device[1], camera_player.resolution[0], camera_player.resolution[1], 1, 5);

		//render_gauss_blur_kernel_launch(camera_player_edge_filter_device[1], camera_player_edge_filter_device[0], camera_player.resolution[0], camera_player.resolution[1], 1);
		//render_gauss_blur_kernel_launch(camera_player_edge_filter_device[1], camera_player_edge_filter_device[0], camera_player.resolution[0], camera_player.resolution[1], 1);

		//render_anti_alias_kernel_launch(camera_player_image_device[0], camera_player_edge_filter_device[0], camera_player_image_device[1], camera_player.resolution[0], camera_player.resolution[1], 3);
		/*
		render_anti_alias_kernel_launch(camera_player_image_device[0], camera_player_image_device[1], camera_player.resolution[0], camera_player.resolution[1], 3, 5.0f);
		render_anti_alias_kernel_launch(camera_player_image_device[1], camera_player_image_device[0], camera_player.resolution[0], camera_player.resolution[1], 3, 5.0f);
		*/

		//cudaMemcpy(camera_player_image[1], camera_player_image_device[0], camera_player.resolution[0] * camera_player.resolution[1] * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		//cudaMemcpy(camera_player_image[0], camera_player_image_device[0], camera_player.resolution[0] * camera_player.resolution[1] * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		/*
		cudaMemcpy(camera_player_edge_filter[0], camera_player_edge_filter_device[0], camera_player.resolution[0] * camera_player.resolution[1] * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaMemcpy(camera_player_edge_filter[1], camera_player_edge_filter_device[1], camera_player.resolution[0] * camera_player.resolution[1] * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		*/
		sdl_update_frame((void*)players_composed, capture_mouse);

		/*
		struct entity* entities = (struct entity*)&l.bf_static.data[l.entities_static_pos];
		entities[0].orientation[2] += 0.2f;

		bit_field_invalidate_bulk(&l.bf_static, l.entities_static_pos, (int)ceilf(sizeof(struct entity)/(float)sizeof(unsigned int)));
		bit_field_update_device(&l.bf_static, 0);
		*/
		/*
		lodepng::encode("output___.png", camera_player_image[1], camera_player.resolution[0], camera_player.resolution[1], LCT_RGB, 8U);
		lodepng::encode("output.png", camera_player_image[0], camera_player.resolution[0], camera_player.resolution[1], LCT_RGB, 8U);
		lodepng::encode("output_.png", camera_player_edge_filter[0], camera_player.resolution[0], camera_player.resolution[1], LCT_GREY, 8U);
		lodepng::encode("output__.png", camera_player_edge_filter[1], camera_player.resolution[0], camera_player.resolution[1], LCT_GREY, 8U);
		*/
		
		long tf_2 = clock();
		long tf_ = tf_2 - tf;
		double td = ((double)tf_ / CLOCKS_PER_SEC);
		sec += td;
		fps++;

		if (sec >= 1.0) {
			printf("main fps: %d, main ft : % f\r\n", fps, tf_ / (double)CLOCKS_PER_SEC);
			sec = 0;
			fps = 0;
		}

		particles_tick();
		cameras = (struct camera*)&bf_dynamic.data[cameras_position_in_bf];

		while (SDL_PollEvent(&sdl_event) != 0) {
			/*
			float camera_delta_z = 0.0f;
				if (sdl_event.type == SDL_MOUSEWHEEL) {
					if (!ui_process_scroll(&bf_rw, mouse_position[0], mouse_position[1], sdl_event.wheel.y)) {
						camera_delta_z -= sdl_event.wheel.y * sensitivity_z;
						camera_move(struct vector3<float>(0.0f, 0.0f, camera_delta_z));
					}
				}

				if (sdl_event.type == SDL_MOUSEMOTION && sdl_event.button.button == SDL_BUTTON(SDL_BUTTON_RIGHT)) {
					float zoom_sensitivity = sensitivity_xy * camera[2] * sensitivity_zoom_ratio;
					if (zoom_sensitivity < 0.2f) zoom_sensitivity = 0.2f;
					camera_move(struct vector3<float>(-sdl_event.motion.xrel * zoom_sensitivity, -sdl_event.motion.yrel * zoom_sensitivity, 0.0f));
					camera_get_crop(camera_crop);
				}
			} else {
				if (sdl_event.type == SDL_MOUSEWHEEL) {
					ui_process_scroll(&bf_rw, mouse_position[0], mouse_position[1], sdl_event.wheel.y);
				}
			}
			*/
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
				struct camera* p = &cameras[player_selected_id];

				if (sdl_event.type == SDL_MOUSEBUTTONDOWN && sdl_event.button.button == SDL_BUTTON(SDL_BUTTON_LEFT)) {
					particle_add(p->position, p->orientation, struct vector3<float>(1.0f, 1.0f, 1.0f));
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
		
			/*
			vector2<unsigned int> current_mouse_game_position = { camera_crop[0] + (unsigned int)(mouse_position[0] * camera[2]), camera_crop[2] + (unsigned int)(mouse_position[1] * camera[2]) };

			if (ui_active != "") {
				if (sdl_event.type == SDL_MOUSEBUTTONUP && sdl_event.button.button == SDL_BUTTON(SDL_BUTTON_LEFT)) {
					//printf("clicking %i %i\n", mouse_position[0], mouse_position[1]);
					bool processed_click = ui_process_click(&bf_rw, mouse_position[0], mouse_position[1]);
					if (processed_click) {
						if (uis[ui_active].active_element_id > -1 && uis[ui_active].ui_elements[uis[ui_active].active_element_id].uet == UET_SCROLLLIST) {
							string ui_element_name = uis[ui_active].ui_elements[uis[ui_active].active_element_id].name;
							if (ui_active == "lobby" && ui_element_name == "maps") {
								if (uis[ui_active].active_element_param > -1) {
									string map_name = map_name_from_index(&bf_rw, uis[ui_active].active_element_param);
									ui_textfield_set_value(&bf_rw, "lobby", "selected_map", map_name.c_str());
									string map_asset_path = "./maps/" + map_name + "_minimap.png";
									ui_value_as_config(&bf_rw, "lobby", "minimap", 0, assets[map_asset_path]);
									ui_value_as_config(&bf_rw, "lobby", "minimap", 1, assets_dimensions[map_asset_path].width);
									ui_value_as_config(&bf_rw, "lobby", "minimap", 2, assets_dimensions[map_asset_path].height);
								}
							}
						}
					}
					else {
						if (map_editor && ui_active == "mapeditor_overlay") {
							mapeditor_process_click();
						}
					}
					players_process_left_click(current_mouse_game_position);
				}
				if (sdl_event.type == SDL_MOUSEBUTTONUP && sdl_event.button.button == 3) {
					players_process_right_click(current_mouse_game_position);
				}
			}
			*/
		}
	}

	return 0;
}