#include "Skeleton.h"

#include <vector>
#include <sstream>
#include <fstream>
#include "Util.h"
#include <iostream>
#include "Entity.h"
#include "M1am.h"
#include "Matrix3.h"
#include "Level.h"

void skeleton_calculate_bone_local_coords(const struct skeleton_bone* bones, unsigned int bone_id, struct vector3<float> t_v, struct vector3<float> *l_v) {
	vector3<float> bone_dir = bones[bone_id].head - bones[bone_id].base;
	vector3<float> bone_dir_u = bone_dir / length(bone_dir);
	vector3<float> v_proj = bones[bone_id].base - -(bone_dir * (dot(t_v - bones[bone_id].base, bone_dir) / dot(bone_dir, bone_dir)));
	float v_proj_dist = length(v_proj - t_v);
	float lambda = 0.0f;
	if (abs(bone_dir_u[0]) > 0.0f) {
		lambda = (v_proj[0] - bones[bone_id].base[0]) / bone_dir_u[0];
	} else if (abs(bone_dir_u[1]) > 0.0f) {
		lambda = (v_proj[1] - bones[bone_id].base[1]) / bone_dir_u[1];
	} else {
		lambda = (v_proj[2] - bones[bone_id].base[2]) / bone_dir_u[2];
	}
	*l_v = {lambda, v_proj_dist, 0.0f};
}

void skeleton_rotate_bone(struct skeleton *sk, unsigned int bone_id, struct vector2<float> theta_phi) {
	struct skeleton_bone* bones = (struct skeleton_bone*)&bf_dynamic.data[sk->bones_position];
	bones[bone_id].theta_phi = theta_phi;
	bit_field_invalidate_bulk(&bf_dynamic, sk->bones_position, (unsigned int)ceilf(2 * sizeof(struct skeleton_bone) / (float)sizeof(unsigned int)));
}

void skeleton_init(string name, unsigned int dyn_entity_id) {
	struct skeleton s;

	entities_dynamic = (struct entity*) &bf_dynamic.data[bf_dynamic_m.entities_dynamic_position_in_bf];
	struct entity* entity_current = &entities_dynamic[dyn_entity_id];

	struct vector3<float>* triangles_static = (struct vector3<float> *) &level_current->bf_static.data[level_current->triangles_static_pos];
	struct vector3<float> *triangles_vecs = &triangles_static[entity_current->triangles_id];

	vector<struct skeleton_bone>	bones;
	vector<struct skeleton_vector>	vectors;
	vectors.resize(entity_current->triangles_c * 3);

	for (int v = 0; v < vectors.size(); v++) {
		vectors[v].bone_id = UINT_MAX;
	}

	stringstream skeleton_path;
	skeleton_path << "models/" << name << ".txt";

	cout << skeleton_path.str() << std::endl;

	vector<string> skeleton_file = util_file_read_lines(skeleton_path.str());

	string comment = "//";
	int b_c = 0;
	for (int t = 0; t < skeleton_file.size(); t++) {
		if (skeleton_file[t].length() == 0) continue;
		if (util_starts_with(skeleton_file[t], comment)) continue;

		vector<string> line_arr = util_split(skeleton_file[t]);

		struct skeleton_bone sb;
		sb.base = { stof(line_arr[0]), stof(line_arr[1]), stof(line_arr[2]) };
		sb.head = { stof(line_arr[3]), stof(line_arr[4]), stof(line_arr[5]) };
		sb.base_orientation = stof(line_arr[6]) * M_PI / 180.0f;

		sb.base_orientation_current = sb.base_orientation;
		sb.head_orientation_current = sb.base_orientation;

		//sb.translation = { 0.0f, 0.0f, 0.0f };

		struct vector3<float> bh_dir = sb.head - sb.base;
		float r = length(bh_dir);

		struct vector3<float> bh_dir_u = bh_dir / r;

		float theta = acos(bh_dir[2] / r);
		float phi = atan2(bh_dir[1], bh_dir[0]);
		printf("bone theta %f, phi %f\n", theta, phi);

		sb.theta_phi = {theta, phi};
		sb.head_rotated = sb.head;

		for (int v = 7; v < line_arr.size(); v++) {
			vector<string> line_arr_sub = util_split(line_arr[v], "-");
			int b = stoi(line_arr_sub[0]);
			int c = b + 1;
			if (line_arr_sub.size() == 2) {
				c = stoi(line_arr_sub[1]);
			}
			for (; b < c; b++) {
				struct vector3<float> vec_current = triangles_vecs[b];
				printf("vec %f %f %f\n", vec_current[0], vec_current[1], vec_current[2]);
				struct vector3<float> lambda_r_phi = { 0.0f, 0.0f, 0.0f };
				skeleton_calculate_bone_local_coords(&sb, 0, vec_current, &lambda_r_phi);
				vec_current = rotate_z(vec_current, -phi);
				vec_current = rotate_x(vec_current, -theta);
				vec_current[0] -= sb.base[0];
				vec_current[1] -= sb.base[1];
				printf("vec_r %f %f %f\n", vec_current[0], vec_current[1], vec_current[2]);
				float phi_v = 0.0f;
				lambda_r_phi[2] = acos(vec_current[0] / lambda_r_phi[1]);
				if (vec_current[1] < 0) {
					lambda_r_phi[2] *= -1.0f;
				}
				printf("bone_id %i, lambda_r_phi %f %f %f\n", b_c, lambda_r_phi[0], lambda_r_phi[1], lambda_r_phi[2]);
				vectors[b].bone_id = b_c;
				vectors[b].lamda_r_phi = lambda_r_phi;
			}
		}
		bones.push_back(sb);
		b_c++;
	}

	unsigned int bones_size_in_mem = bones.size() * sizeof(struct skeleton_bone);
	unsigned int bones_size_in_bf = (unsigned int)ceilf(bones_size_in_mem/(float) sizeof(unsigned int));
	s.bones_position = bit_field_add_bulk(&bf_dynamic, (unsigned int *)bones.data(), bones_size_in_bf, bones_size_in_mem)+1;

	unsigned int vectors_size_in_mem = vectors.size() * sizeof(struct skeleton_bone);
	unsigned int vectors_size_in_bf = (unsigned int)ceilf(vectors_size_in_mem/(float)sizeof(unsigned int));
	s.vectors_position = bit_field_add_bulk(&bf_dynamic, (unsigned int *)vectors.data(), vectors_size_in_bf, vectors_size_in_mem) + 1;

	unsigned int skeleton_size_in_mem = sizeof(struct skeleton);
	unsigned int skeleton_size_in_bf = (unsigned int)ceilf(skeleton_size_in_mem / (float)sizeof(unsigned int));
	unsigned int skeleton_position = bit_field_add_bulk(&bf_dynamic, (unsigned int *) &s, skeleton_size_in_bf, skeleton_size_in_mem) + 1;

	entities_dynamic = (struct entity*)&bf_dynamic.data[bf_dynamic_m.entities_dynamic_position_in_bf];
	entity_current = &entities_dynamic[dyn_entity_id];
	entity_current->skeleton_id = skeleton_position;
}