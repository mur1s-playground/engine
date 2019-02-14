#include "Framebuffer.hpp"

#include "stdlib.h"
#include "Camera.hpp"
#include "string.h"

void framebuffer_allocate(struct framebuffer *fb, struct camera *cameras, int cameras_c, int len, unsigned int device_id) {
	fb->device_frames = (unsigned char **) malloc(len*sizeof(unsigned char *));
	fb->host_frames = (unsigned char **) malloc(len*sizeof(unsigned char *));

	#ifdef _WIN32
	fb->locks = (HANDLE *) malloc(len*sizeof(HANDLE *));
	#else
	fb->locks = (pthread_mutex_t *) malloc(len*sizeof(pthread_mutex_t));
	#endif
	

	long total_size = 0;
	for (int j = 0; j < cameras_c; j++) {
		total_size += (cameras[j].resolution[0] * cameras[j].resolution[1]);
	}
	total_size *= 4;

	for (int i = 0; i < len; i++) {
		fb->device_frames[i] = camera_allocate_image_on_device(total_size, device_id);
		fb->host_frames[i] = (unsigned char *) malloc(total_size*sizeof(unsigned char));
		#ifdef _WIN32
		fb->locks[i] = CreateMutex(NULL, FALSE, NULL);
		#else
		pthread_mutex_init(&fb->locks[i], NULL);
		#endif
	}
	fb->len = len;
}

void framebuffer_allocate_splitscreen(struct framebuffer *fb, struct camera *camera, int len, vector2<unsigned int> device_ids) {
	fb->device_frames = (unsigned char **) malloc(2*len*sizeof(unsigned char *));
	fb->host_frames = (unsigned char **) malloc(len*sizeof(unsigned char *));

        #ifdef _WIN32
        fb->locks = (HANDLE *) malloc(len*sizeof(HANDLE *));
        #else
        fb->locks = (pthread_mutex_t *) malloc(len*sizeof(pthread_mutex_t));
        #endif

	long size = (camera->resolution[0]*camera->resolution[1]);
	for (int i = 0; i < len; i++) {
		fb->device_frames[(2*i)]   = camera_allocate_image_on_device(size/2, device_ids[0]);
		fb->device_frames[(2*i)+1] = camera_allocate_image_on_device(size/2, device_ids[1]);
	}
	for (int i = 0; i < len; i++) {
		fb->host_frames[i] = (unsigned char *) malloc(size*4*sizeof(unsigned char));
		memset(fb->host_frames[i], 0, size*4*sizeof(unsigned char));
                #ifdef _WIN32
                fb->locks[i] = CreateMutex(NULL, FALSE, NULL);
                #else
                pthread_mutex_init(&fb->locks[i], NULL);
                #endif
	}
	fb->len = len;
}
