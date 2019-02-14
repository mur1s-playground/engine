#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <cuda_runtime.h>
#include "Vector2.hpp"
#include "Vector3.hpp"
#include "Framebuffer.hpp"
#include "World.hpp"

struct camera {
	struct vector3<float>	position;
	struct vector3<float>	orientation;

	struct vector2<float>	fov;
	struct vector2<int>	resolution;
};

struct camera_segment {
	int phi_start;
	int phi_end;
	int theta_start;
	int theta_end;
};

unsigned char *camera_allocate_image_on_device(long size, unsigned int device_id);

void camera_render_image(struct world *w, unsigned char *d_image, unsigned char *h_image, unsigned int device_id, unsigned int camera_id, unsigned int phi_start, unsigned int phi_end, unsigned int theta_start, unsigned int theta_end);

void camera_render_image_splitscreen(struct world *w, struct framebuffer *fb, unsigned int fb_id, unsigned int fb_width, unsigned int fb_height, vector2<int> device_ids, unsigned int camera_id);

//vector3<float> camera_get_direction(const struct camera *cam, const int phi_x, const int theta_y);

#endif /* CAMERA_HPP */
