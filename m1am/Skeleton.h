#pragma once

#include "Vector3.h"
#include "Vector2.h"
#include <string>

using namespace std;

struct skeleton {
	unsigned int	bones_position;
	unsigned int	vectors_position;
};

struct skeleton_vector {
	unsigned int	bone_id;
	struct vector3<float> lamda_r_phi;
};

struct skeleton_bone {
	struct vector3<float>	base;
	struct vector3<float>	head;
	float					base_orientation;

	//struct vector3<float>	translation;
	struct vector2<float>	theta_phi;

	//
	struct vector3<float>	head_rotated;
	float					base_orientation_current;
	float					head_orientation_current;
};

void skeleton_rotate_bone(struct skeleton* sk, unsigned int bone_id, struct vector2<float> theta_phi);
void skeleton_init(string name, unsigned int dyn_entity_id);
