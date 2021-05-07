#include "Camera.h"
#include "M1am.h"
#include "Util.h"

struct camera*	cameras						= nullptr;

unsigned int	cameras_c					= 0;
unsigned int	cameras_size_in_bf			= UINT_MAX;
unsigned int	cameras_position_in_bf		= UINT_MAX;

void camera_move(unsigned int camera_id, vector3<float> position_d, vector3<float> orientation_d) {
	cameras = (struct camera*)&bf_dynamic.data[cameras_position_in_bf];
	struct camera* p = &cameras[camera_id];

	p->orientation = p->orientation - -(orientation_d);
	
	if (p->orientation[0] < 0) p->orientation[0] = 0.0f;
	if (p->orientation[0] > M_PI) p->orientation[0] = M_PI;
	//if (length(orientation_d) > 0.0f) {
		float phi_start = -p->fov[0] / 2.0f;
		float phi_end = p->fov[0] / 2.0f;

		float theta_start = -p->fov[1] / 2.0f;
		float theta_end = p->fov[1] / 2.0f;

		p->camera_ray_orientation_lt = {
				sinf(theta_start + p->orientation[0]) * -cosf(phi_start + p->orientation[2]),
				sinf(theta_start + p->orientation[0]) * sinf(phi_start + p->orientation[2]),
				cosf(theta_start + p->orientation[0]),
		};

		struct vector3<float> camera_ray_orientation_rt = {
			sinf(theta_start + p->orientation[0]) * -cosf(phi_end + p->orientation[2]),
			sinf(theta_start + p->orientation[0]) * sinf(phi_end + p->orientation[2]),
			cosf(theta_start + p->orientation[0]),
		};

		p->camera_ray_orientation_dt = camera_ray_orientation_rt - p->camera_ray_orientation_lt;

		p->camera_ray_orientation_lb = {
				sinf(theta_end +  p->orientation[0]) * -cosf(phi_start +  p->orientation[2]),
				sinf(theta_end +  p->orientation[0]) * sinf(phi_start +  p->orientation[2]),
				cosf(theta_end +  p->orientation[0]),
		};

		struct vector3<float> camera_ray_orientation_rd = {
			sinf(theta_end +  p->orientation[0]) * -cosf(phi_end +  p->orientation[2]),
			sinf(theta_end +  p->orientation[0]) * sinf(phi_end +  p->orientation[2]),
			cosf(theta_end +  p->orientation[0]),
		};

		p->camera_ray_orientation_db = camera_ray_orientation_rd - p->camera_ray_orientation_lb;

	//}

	float o_s = sinf(-p->orientation[2] + M_PI / 2.0);
	float o_c = cosf(-p->orientation[2] + M_PI / 2.0);

	float dx = position_d[0];
	float dy = position_d[1];

	struct vector2<float> move_rot = {
		(dx * o_c - dy * o_s),
		(dx * o_s + dy * o_c)
	};

	p->position = {
			p->position[0] + move_rot[0],
			p->position[1] + move_rot[1],
			p->position[2] + position_d[2]
	};
	unsigned int camera_position_in_bf_start = (unsigned int)floor(camera_id * cameras_size_in_bf/cameras_c);
	unsigned int camera_position_in_bf_end = (unsigned int)ceil((camera_id + 1) * cameras_size_in_bf / cameras_c);
	bit_field_invalidate_bulk(&bf_dynamic, cameras_position_in_bf + camera_position_in_bf_start, camera_position_in_bf_end - camera_position_in_bf_start);
}