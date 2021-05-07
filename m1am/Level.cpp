#include "Level.h"

#include "Util.h"
#include "Grid.h"
#include "M1am.h"

#include <sstream>
#include <map>
#include "Vector3.h"
#include "Entity.h"
#include "lodepng.h"

#include <iostream>

struct level* level_current	= nullptr;

void level_load(struct level* level, string name) {
	entity_dynamic_free_clear();

	stringstream levelmeta_path;
	levelmeta_path << "levels/" << name << ".level";

	size_t bin_l = 0;
	util_read_binary(levelmeta_path.str(), (unsigned char*)level, &bin_l);
	memset(&level->bf_static, 0, sizeof(struct bit_field));

	stringstream levelpath;
	levelpath << "levels/" << name << ".bf";

	cout << "loading bf" << std::endl;
	bit_field_load_from_disk(&level->bf_static, levelpath.str());

	cout << "entities_pos " << level->entities_static_pos << std::endl;
	cout << "triangles_pos " << level->triangles_static_pos << std::endl;
	cout << "textures_map_pos " << level->textures_map_static_pos << std::endl;
	cout << "textures_pos " << level->textures_static_pos << std::endl;

	bit_field_register_device(&level->bf_static, 0);
	bit_field_update_device(&level->bf_static, 0);

	level_current = level;

	stringstream levelpath_dyn;
	levelpath_dyn << "levels/" << name << "_dynamic.bf";

	bit_field_load_from_disk(&bf_dynamic, levelpath_dyn.str());

	stringstream levelpath_dyn_meta;
	levelpath_dyn_meta << "levels/" << name << "_dynamic.meta";

	util_read_binary(levelpath_dyn_meta.str(), (unsigned char*)&bf_dynamic_m, &bin_l);

	cout << "e_grid pos " << bf_dynamic_m.entity_grid_position_in_bf << std::endl;
	cout << "ent pos " << bf_dynamic_m.entities_dynamic_position_in_bf << std::endl;
	cout << "dyn alloc count " << bf_dynamic_m.entities_dynamic_allocated_count << std::endl;

	bit_field_register_device(&bf_dynamic, 0);
	bit_field_update_device(&bf_dynamic, 0);

	entity_dynamic_free_preallocate(bf_dynamic_m.entities_dynamic_allocated_count);
}

void level_save(struct level *level, string name) {
	bit_field_init(&level->bf_static, 128, 1024);
	bit_field_init(&bf_dynamic, 128, 1024);

	stringstream level_def_path;
	level_def_path << "levels/" << name << ".txt";

	vector<string> level_def = util_file_read_lines(level_def_path.str());

	cout << "level def loaded" << std::endl;

	vector<struct entity>						entities_static = vector<struct entity>();

	vector<struct vector3<float>>				triangles_static = vector<struct vector3<float>>();
	map<string, unsigned int>					triangles_static_id = map<string, unsigned int>();
	map<unsigned int, unsigned int>				triangles_static_counts = map<unsigned int, unsigned int>();
	map<string, unsigned int>					triangles_static_grid_id = map<string, unsigned int>();
	map<string, float>							triangles_static_radius = map<string, float>();

	vector<string>								textures_map_file_static = vector<string>();
	map<string, unsigned int>					textures_map_file_static_id = map<string,unsigned int>();
	
	map<string, unsigned int>					textures_map_static_id = map<string, unsigned int>();
	vector<struct triangle_texture>				textures_map_static = vector<struct triangle_texture>();

	map<string, unsigned int>					textures_static_id = map<string, unsigned int>();
	map<string, struct vector2<unsigned int>>	textures_static_dimensions = map<string, struct vector2<unsigned int>>();
	vector<unsigned char>						textures_static = vector<unsigned char>();

	string comment = "//";

	struct vector3<float>						entities_static_min_coords = { FLT_MAX, FLT_MAX, FLT_MAX };
	struct vector3<float>						entities_static_max_coords = { FLT_MIN, FLT_MIN, FLT_MIN };

	for (int l = 0; l < level_def.size(); l++) {
		if (level_def[l].length() == 0) continue;
		if (util_starts_with(level_def[l], comment)) continue;
		vector<string> arr = util_split(level_def[l]);
		
		struct entity e;
		e.position = { stof(arr[4]),stof(arr[5]), stof(arr[6]) };
		e.orientation = { stof(arr[7]) * M_PI / 180.0f, stof(arr[8]) * M_PI / 180.0f, stof(arr[9]) * M_PI / 180.0f };
		e.scale = stof(arr[10]);
		e.skeleton_id = UINT_MAX;

		//--------------//
		//LOAD TRIANGLES//
		//--------------//
		unsigned int triangles_c = UINT_MAX;
		if (strcmp(arr[0].c_str(), "static") == 0) {
			map<string, unsigned int>::iterator triangles_static_id_it = triangles_static_id.find(arr[1]);

			if (triangles_static_id_it != triangles_static_id.end()) {
				unsigned int triangles_id = triangles_static_id_it->second;
				e.triangles_id = triangles_id;
				e.triangles_c = triangles_static_counts.find(triangles_id)->second;

				map<string, unsigned int>::iterator triangles_static_grid_id_it = triangles_static_grid_id.find(arr[1]);
				if (triangles_static_grid_id_it != triangles_static_grid_id.end()) {
					e.triangles_grid_id = triangles_static_grid_id_it->second;
				}

				map<string, float>::iterator triangles_static_radius_it = triangles_static_radius.find(arr[1]);
				if (triangles_static_radius_it != triangles_static_radius.end()) {
					e.radius = triangles_static_radius_it->second;
				}

				triangles_c = e.triangles_c;
			} else {

				stringstream triangles_path;
				triangles_path << "models/" << arr[1] << ".txt";

				cout << triangles_path.str() << std::endl;

				vector<string> triangles = util_file_read_lines(triangles_path.str());

				unsigned int triangles_id = triangles_static.size();
				triangles_static_id.insert(pair<string, unsigned int>(arr[1], triangles_id));

				unsigned int count = 0;

				float max_radius = 0.0f;

				struct grid triangle_grid;

				for (int t = 0; t < triangles.size(); t++) {
					if (triangles[t].length() == 0) continue;
					if (util_starts_with(triangles[t], comment)) continue;

					cout << "getting vector3 from " << triangles[t] << std::endl;
					struct vector3<float> tmp = util_get_vector3<float>(triangles[t]);

					if (length(tmp) > max_radius) {
						max_radius = length(tmp);
					}

					cout << count << " " <<  tmp[0] << " " << tmp[1] << " " << tmp[2] << std::endl;

					triangles_static.push_back(tmp);
					count++;
				}

				e.radius = max_radius;
				triangles_static_radius.insert(pair<string, float>(arr[1], max_radius));
				printf("radius %f\n", max_radius);

				grid_init(&level->bf_static, &triangle_grid, struct vector3<float>(2.0f * max_radius, 2.0f * max_radius, 2.0f * max_radius), struct vector3<float>(0.25f, 0.25f, 0.25f), struct vector3<float>(max_radius, max_radius, max_radius), 0);
				cout << "initialized triangles static grid" << std::endl;
				int tr_c = 0;
				count = 0;
				for (int t = 0; t < triangles.size(); t++) {
					if (triangles[t].length() == 0) continue;
					if (util_starts_with(triangles[t], comment)) continue;

					cout << t << " " <<triangles[t] << std::endl;
					struct vector3<float> tmp = util_get_vector3<float>(triangles[t]);

					vector3<float> max, min;
					for (int d = 0; d < 3; d++) {
						if (tr_c == 0) {
							max[d] = tmp[d];
							min[d] = tmp[d];
						} else {
							if (max[d] < tmp[d]) max[d] = tmp[d];
							if (min[d] > tmp[d]) min[d] = tmp[d];
						}
					}
					tr_c++;

					if (tr_c == 3) {
						cout << "adding grid object " << std::endl;
						grid_object_add(&level->bf_static, level->bf_static.data, triangle_grid.position_in_bf, struct vector3<float>(0.0f, 0.0f, 0.0f), struct vector3<float>(1.0f, 1.0f, 1.0f), min, max, count / 3);
						tr_c = 0;
					}
					count++;
				}
				e.triangles_grid_id = triangle_grid.position_in_bf;
				triangles_static_grid_id.insert(pair<string, unsigned int>(arr[1], e.triangles_grid_id));


				for (int d = 0; d < 3; d++) {
					if (e.position[d] - e.radius < entities_static_min_coords[d]) {
						entities_static_min_coords[d] = e.position[d] - (e.radius * e.scale);
					}
					if (e.position[d] + e.radius > entities_static_max_coords[d]) {
						entities_static_max_coords[d] = e.position[d] + (e.radius * e.scale);
					}
				}

				triangles_static_counts.insert(pair<unsigned int, unsigned int>(triangles_id, count/3));
				e.triangles_id = triangles_id;
				e.triangles_c = count/3;
				triangles_c = count/3;
			}
		} else {
			//handle dynamic triangles
		}

		cout << "starting load texture mapping file" << std::endl;

		//-----------------------------------//
		//LOAD TRIANGLES TEXTURE MAPPING FILE//
		//-----------------------------------//
		map<string, unsigned int>::iterator textures_map_file_static_id_it = textures_map_file_static_id.find(arr[3]);
		unsigned int texture_map_file_id = UINT_MAX;
		if (textures_map_file_static_id_it != textures_map_file_static_id.end()) {
			texture_map_file_id = textures_map_file_static_id_it->second;
		} else {
			stringstream texture_map_path;
			texture_map_path << "models/" << arr[1] << "_" << arr[3] << ".txt";

			cout << texture_map_path.str() << std::endl;

			vector<string> texture_map = util_file_read_lines(texture_map_path.str());
			bool start_id = true;
			for (int m = 0; m < texture_map.size(); m++) {
				if (texture_map[m].length() == 0) continue;
				if (util_starts_with(texture_map[m], comment)) continue;
				if (start_id) {
					textures_map_file_static_id.insert(pair<string, unsigned int>(arr[3], textures_map_file_static.size()));
					texture_map_file_id = textures_map_file_static.size();
					start_id = false;
				}
				textures_map_file_static.push_back(texture_map[m]);
			}
		}
		
		//----------------//
		//LOAD TEXTURE MAP//
		//----------------//
		map<string, unsigned int>::iterator textures_map_static_id_it = textures_map_static_id.find(arr[3]);
		if (strcmp(arr[2].c_str(), "static") == 0 && textures_map_static_id_it != textures_map_static_id.end()) {
			e.texture_map_id = textures_map_static_id_it->second;
		} else {
			if (strcmp(arr[2].c_str(), "static") == 0) {
				e.texture_map_id = textures_map_static.size();
				cout << "texture map id " << e.texture_map_id << std::endl;
				textures_map_static_id.insert(pair<string, unsigned int>(arr[3], e.texture_map_id));
			} else {
				//handle dynamic texture map id
			}

			cout << texture_map_file_id << " +" << triangles_c << std::endl;

			for (int m = texture_map_file_id; m < texture_map_file_id + triangles_c; m++) {

				cout << textures_map_file_static[m] << std::endl;

				struct triangle_texture tt;
				
				vector<string> texture_map_arr = util_split(textures_map_file_static[m]);

				if (strcmp(texture_map_arr[0].c_str(), "static") == 0) {
					map<string, unsigned int>::iterator texture_static_id_it = textures_static_id.find(texture_map_arr[1]);
					if (texture_static_id_it != textures_static_id.end()) {
						tt.texture_id = texture_static_id_it->second;
					} else {
						stringstream texture_filename;
						texture_filename << "textures/" << texture_map_arr[1] << ".png";

						std::vector<unsigned char> image;
						unsigned int width = 0;
						unsigned int height = 0;
						unsigned int size = 0;
						lodepng::decode(image, width, height, texture_filename.str(), LCT_RGBA, 8U);

						//prepend image metadata to the image vector
						for (int i = 0; i < 8; i++) {
							image.insert(image.begin(), 0);
						}
						unsigned int* im_w = (unsigned int*) image.data();
						*im_w = width;
						unsigned int* im_h = (unsigned int*) (image.data() + 4);
						*im_h = height;
						//

						unsigned int start_idx = textures_static.size();
						textures_static_id.insert(pair<string, unsigned int>(texture_map_arr[1], start_idx));
						textures_static.resize(textures_static.size() + image.size());
						cout << texture_filename.str() << " " << width << "x" << height << " " << image.size() <<std::endl;
						memcpy(textures_static.data() + start_idx, image.data(), image.size());
						textures_static_dimensions.insert(pair<string, struct vector2<unsigned int>>(texture_map_arr[1], struct vector2<unsigned int>(width, height)));
						tt.texture_id = start_idx;
					}
				} else {
					//handle dynamic texture
				}

				tt.texture_coord = { stoi(texture_map_arr[2]), stoi(texture_map_arr[3]) };
				tt.texture_orientation = stof(texture_map_arr[4]) * M_PI / 180.0f;
				tt.texture_scale = stof(texture_map_arr[5]);
				textures_map_static.push_back(tt);
			}
		}

		cout << l << " pushing" << std::endl;
		entities_static.push_back(e);
	}

	struct grid entity_grid;

	struct vector3<float> entity_grid_dimensions = entities_static_max_coords - entities_static_min_coords;

	//NOTE: center is not the actual center, but the translation of the min_coords to {0, 0, 0}
	struct vector3<float> entity_grid_center = -entities_static_min_coords;

	//grid_init(&level->bf_static, &entities_static_grid, entities_static_grid_dimensions, struct vector3<float>(1.0f, 1.0f, 1.0f), entities_static_grid_center);
	grid_init(&bf_dynamic, &entity_grid, struct vector3<float>(10.0f, 10.0f, 10.0f), struct vector3<float>(1.0f, 1.0f, 1.0f), struct vector3<float>(5.0f, 5.0f, 5.0f), 0);
	for (int e = 0; e < entities_static.size(); e++) {
		struct vector3<float> e_radius = { entities_static[e].radius, entities_static[e].radius, entities_static[e].radius };
		struct vector3<float> s_radius = e_radius * entities_static[e].scale;

		cout << "s_radius: " << s_radius[0] << std::endl;
		
		entities_static[e].radius = s_radius[0];
		if (s_radius[1] > entities_static[e].radius)entities_static[e].radius = s_radius[1];
		if (s_radius[2] > entities_static[e].radius)entities_static[e].radius = s_radius[2];
		
		grid_object_add(&bf_dynamic, bf_dynamic.data, entity_grid.position_in_bf, entities_static[e].position, struct vector3<float>(entities_static[e].scale, entities_static[e].scale, entities_static[e].scale), -e_radius, e_radius, e);

		cout << "grid index: " << grid_get_index(bf_dynamic.data, entity_grid.position_in_bf, entities_static[e].position) << std::endl;
	}
	level->entities_static_count = entities_static.size();
	grid_postallocate(&bf_dynamic, &entity_grid, 16);
	bf_dynamic_m.entity_grid_position_in_bf = entity_grid.position_in_bf;
	cout << "entity_grid_pos: " << bf_dynamic_m.entity_grid_position_in_bf << std::endl;

	unsigned int entities_static_size_in_mem = entities_static.size() * sizeof(struct entity);
	unsigned int entities_static_size_in_bf = (unsigned int)ceilf(entities_static_size_in_mem / (float)sizeof(unsigned int));
	unsigned int entities_static_pos = bit_field_add_bulk(&level->bf_static, (unsigned int *)entities_static.data(), entities_static_size_in_bf, entities_static_size_in_mem);
	cout << "entities_size " << entities_static_size_in_mem << std::endl;
	level->entities_static_pos = entities_static_pos + 1;

	unsigned int triangles_static_size_in_mem = triangles_static.size() * sizeof(struct vector3<float>);
	unsigned int triangles_static_size_in_bf = (unsigned int)ceilf(triangles_static_size_in_mem / (float)sizeof(unsigned int));
	unsigned int triangles_static_pos = bit_field_add_bulk(&level->bf_static, (unsigned int*)triangles_static.data(), triangles_static_size_in_bf, triangles_static_size_in_mem);
	cout << "triangles_size " << triangles_static_size_in_mem << std::endl;
	level->triangles_static_pos = triangles_static_pos + 1;

	unsigned int textures_map_static_size_in_mem = textures_map_static.size() * sizeof(struct triangle_texture);
	unsigned int textures_map_static_size_in_bf = (unsigned int)ceilf(textures_map_static_size_in_mem / (float)sizeof(unsigned int));
	unsigned int textures_map_static_pos = bit_field_add_bulk(&level->bf_static, (unsigned int*)textures_map_static.data(), textures_map_static_size_in_bf, textures_map_static_size_in_mem);
	cout << "textures_map_size " << textures_map_static_size_in_mem << std::endl;
	level->textures_map_static_pos = textures_map_static_pos + 1;
	
	unsigned int textures_static_size_in_mem = textures_static.size() * sizeof(unsigned char);
	unsigned int textures_static_size_in_bf = (unsigned int)ceilf(textures_static_size_in_mem / (float)sizeof(unsigned int));
	unsigned int textures_static_pos = bit_field_add_bulk(&level->bf_static, (unsigned int*)textures_static.data(), textures_static_size_in_bf, textures_static_size_in_mem);
	cout << "textures_size " << textures_static_size_in_mem << std::endl;
	level->textures_static_pos = textures_static_pos + 1;

	entity_dynamic_preallocate(100);

	stringstream levelpath;
	levelpath << "levels/" << name << ".bf";

	cout << "saving bf" << std::endl;
	bit_field_save_to_disk(&level->bf_static, levelpath.str());

	stringstream levelmeta_path;
	levelmeta_path << "levels/" << name << ".level";

	util_write_binary(levelmeta_path.str(), (unsigned char *)level, sizeof(struct level));

	stringstream levelpath_dyn;
	levelpath_dyn << "levels/" << name << "_dynamic.bf";

	bit_field_save_to_disk(&bf_dynamic, levelpath_dyn.str());

	stringstream levelpath_dyn_meta;
	levelpath_dyn_meta << "levels/" << name << "_dynamic.meta";

	util_write_binary(levelpath_dyn_meta.str(), (unsigned char*)&bf_dynamic_m, sizeof(struct bf_dynamic_meta));
}