#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "Vector2.hpp"

struct framebuffer {
	unsigned char **device_frames;
	unsigned char **host_frames;

	#ifdef _WIN32
	HANDLE *locks;
	#else
	pthread_mutex_t *locks;
	#endif

	int len;
};

void framebuffer_allocate(struct framebuffer *fb, struct camera *cameras, int cameras_c, int len, unsigned int device_id);
void framebuffer_allocate_splitscreen(struct framebuffer *fb, struct camera *camera, int len, vector2<unsigned int> device_ids);


#endif /* FRAMEBUFFER_H */
