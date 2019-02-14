#include "Camera.hpp"

#include <stdlib.h>
#include <stdio.h>
#include "Render.hpp"

unsigned char *camera_allocate_image_on_device(long size, unsigned int device_id) {
	cudaError_t err = cudaSuccess;

	//printf("total_size %lu\r\n", total_size);
	unsigned char *device_images = NULL;
	err = cudaSetDevice(device_id);
	err = cudaMalloc((void **)&device_images, size*4*sizeof(unsigned char));
	if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device images (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	cudaDeviceSynchronize();
	return device_images;
}

unsigned char *camera_download_device_image_into_buffer(unsigned char *d_image, unsigned char *h_image, unsigned int device_id, unsigned int phi_start, unsigned int phi_end, unsigned int theta_start, unsigned int theta_end) {
	//printf("total_size %lu\r\n", total_size);
	cudaError_t err = cudaSuccess;
	err = cudaSetDevice(device_id);
	err = cudaMemcpy(h_image, d_image, (phi_end-phi_start)*(theta_end-theta_start)*4*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
	    fprintf(stderr, "Failed to download device image!\n");
        exit(EXIT_FAILURE);
	}
	return h_image;
}

void camera_render_image(struct world *w, unsigned char *d_image, unsigned char *h_image, unsigned int device_id, unsigned int camera_id, unsigned int phi_start, unsigned int phi_end, unsigned int theta_start, unsigned int theta_end) {
	launch_calc_pixel(w, d_image, device_id, camera_id, phi_start, phi_end, theta_start, theta_end);

	camera_download_device_image_into_buffer(d_image, h_image, device_id, phi_start, phi_end, theta_start, theta_end);
}

void camera_render_image_splitscreen(struct world *w, struct framebuffer *fb, unsigned int fb_id, unsigned int fb_width, unsigned int fb_height, vector2<int> device_ids, unsigned int camera_id) {
	launch_calc_pixel(w, fb->device_frames[(fb_id*2)+1], device_ids[1], camera_id,  0, fb_width, fb_height/2, fb_height);
	launch_calc_pixel(w, fb->device_frames[(fb_id*2)], device_ids[0], camera_id, 0, fb_width, 0, fb_height/2);

	camera_download_device_image_into_buffer(fb->device_frames[(fb_id*2)+1], &(fb->host_frames[fb_id])[4*fb_width*(fb_height/2)], device_ids[1], 0, fb_width, fb_height/2, fb_height);
	camera_download_device_image_into_buffer(fb->device_frames[(fb_id*2)], fb->host_frames[fb_id], device_ids[0], 0, fb_width, 0, fb_height/2);
}

/*
vector3<float> camera_get_direction(const struct camera *cam, const int phi_x, const int theta_y) {
	vector3<float> dir;

	vector2<int> resolution = cam->resolution;
	vector3<float> orientation = cam->orientation;
	vector2<float> fov = cam->fov;

        float phi = -fov[0]/2.0 + ((phi_x) * (fov[0]/(float)resolution[0]));
        float theta = -fov[1]/2.0 + ((theta_y) * (fov[1]/(float)resolution[1]));

		#ifdef _WIN32
		float M_PI = 3.14159265358979323846;
		#endif

        dir[0] = -(sinf(theta*M_PI/(2.0*90.0) + M_PI/2.0) * cosf(phi*M_PI/(2.0*90) + M_PI/2.0));
        dir[1] = (sinf(theta*M_PI/(2.0*90.0) + M_PI/2.0) * sinf(phi*M_PI/(2.0*90) + M_PI/2.0));
        dir[2] = (cosf(theta*M_PI/(2.0*90.0) + M_PI/2.0));
        dir = rotate_x(dir, orientation[0], sin_f);
        dir = rotate_y(dir, orientation[1]);
        dir = rotate_z(dir, orientation[2]);

	return dir;
}
*/
