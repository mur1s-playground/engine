#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif


#include "Framebuffer.hpp"
#include "World.hpp"
//#include "encoder/lodepng.h"
#include "SDLShow.hpp"
#include <time.h>
#include <string.h>
#include <math.h>
#include "EntityGrid.hpp"
#include "Catalog.hpp"

int main(int argc, char **argv) {
	struct world w1;
	world_init(&w1);

	struct camera c1;
//	c1.position = { -1.0, -5.0, 1.0};
//	c1.orientation = { 0.0, 0.0, 0.0};
	c1.position = { 10.0, 51.0, 2.5};
	c1.orientation = { 90.0, 0.0, 0.0};
	c1.fov = { 160.0, 90.0 };
	c1.resolution = { 1280, 720 };

	struct camera c2;
	c2.position = { -5.0, -5.0, 2.5};
	c2.orientation = { 20.0, 0.0, 10.0 };
	c2.fov = { 160.0, 90.0 };
	c2.resolution = { 1920, 1080 };

	world_add_camera(&w1, &c1);
//	world_add_camera(&w1, &c2);


	struct texture_mapper tm;
	texture_mapper_init(&tm);

	unsigned char *default_texture = entity_generate_default_texture(100, 100, 255, 255, 0, 0);

	unsigned int default_texture_id = texture_mapper_texture_add(&tm, default_texture, 100, 100);

	unsigned int tex_counter = 0;

	/////////////////////////
	// GERMANY MAP EXAMPLE //
	/////////////////////////
	DIR *dirp;
	struct dirent *dp;
	dirp = opendir("catalog/zipcode_complete/");
//	dirp = opendir("catalog/zipcode_minmax_2drect/");
	do {
		if ((dp = readdir(dirp)) != NULL) {
			char buf[256];
			sprintf(buf, "catalog/zipcode_complete/%s", dp->d_name);
//			sprintf(buf, "catalog/zipcode_minmax_2drect/%s", dp->d_name);
			float rad = 0;
                        unsigned int rect2d_vecs_pos = catalog_load_vectors_into_bit_field(buf, &w1.triangles, &rad);
                        if (rect2d_vecs_pos != 0) {
                                float *rect2d_pos = catalog_load_position(buf);
                                struct entity rect2d;
                                entity_init(&rect2d, rect2d_vecs_pos, rad);
                                for (int i = 0; i < 3; i++) {
                                        rect2d.position[i] = rect2d_pos[i];
                                }
				unsigned char *tex = entity_generate_default_texture(100, 100, 255, 123*(tex_counter++)%256, 234*(tex_counter)%256, 123*(tex_counter)%256);
				unsigned int tex_id_t = texture_mapper_texture_add(&tm, tex, 100, 100);

                                rect2d.texture_id = tex_id_t;
                                world_add_entity(&w1, &rect2d);
                                free(rect2d_pos);
                        }

		}
	} while (dp != NULL);

	//////////////////////////
	//	CUBE EXAMPLE	//
	//////////////////////////
/*
	unsigned int cube_vectors_c = 0;
	float *cube_vectors = entity_generate_cube_triangles(&cube_vectors_c);

	unsigned int cube_vectors_pos = bit_field_add_bulk(&w1.triangles, (unsigned int *) cube_vectors, ceil(cube_vectors_c*sizeof(float)/(float)sizeof(unsigned int)));

	for (int i = 0; i < 10; i++) {
		struct entity *cube = entity_generate_cube(cube_vectors_pos);
		cube->position[0] = -10 + 1.5*i;
		cube->texture_id = default_texture_id;
		world_add_entity(&w1, cube);
	}

	unsigned char *default_texture_2 = entity_generate_default_texture(50, 50, 127, 255, 123, 123);
	default_texture_id = texture_mapper_texture_add(&tm, default_texture_2, 50, 50);

	for (int i = 0; i < 10; i++) {
		struct entity *cube = entity_generate_cube(cube_vectors_pos);
		cube->position[0] = -10 + 1.5*i;
		cube->position[2] = -2;
		cube->texture_id = default_texture_id;
		world_add_entity(&w1, cube);
	}

	unsigned char *default_texture_3 = entity_generate_default_texture(25, 25, 40, 123, 123, 213);
	default_texture_id = texture_mapper_texture_add(&tm, default_texture_3, 25, 25);

	for (int i = 0; i < 10; i++) {
		struct entity *cube = entity_generate_cube(cube_vectors_pos);
		cube->position[0] = -10 + 1.5*i;
		cube->position[1] = -2;
		cube->position[2] = -2;
		cube->texture_id = default_texture_id;
		world_add_entity(&w1, cube);
	}
*/
//	bit_field_dump(&w1.data);
//	printf("%i\r\n", w1.tm->textures.pages);
//	bit_field_dump(&tm.textures);
//	bit_field_dump(&w1.triangles);
//	bit_field_dump(&w1.eg.data);
//	entity_grid_dump(&w1.eg);

	int fb_len = 2;
	struct framebuffer fb;
	framebuffer_allocate_splitscreen(&fb, &c1, fb_len, {0, 1});

//	framebuffer_allocate(&fb, &c1, 1, fb_len, 0);
//	framebuffer_allocate(&fb2, &c1, 1, fb_len, 1);

	texture_mapper_register_device(&tm, 0);
	texture_mapper_register_device(&tm, 1);

	texture_mapper_update_device(&tm, 0);
	texture_mapper_update_device(&tm, 1);

	world_set_texture_mapper(&w1, &tm);

	world_register_device(&w1, 0);
	world_register_device(&w1, 1);

	bit_field_update_device(&w1.triangles, 0);
	bit_field_update_device(&w1.triangles, 1);

	#ifdef _WIN32
	HANDLE sdl_thread;
	sdl_thread = CreateThread(NULL, 0, sdl_show_loop, &fb, 0, NULL);
	#else
	pthread_t sdl_thread;
	pthread_create(&sdl_thread, NULL, sdl_show_loop, &fb);
	#endif

	int fps = 0;
	double sec = 0;

	int sh = 1;
	int once = 0;
	float harr = 0;
	while (1) {
		for (int i = 0; i < fb_len; i++) {
			long tf = clock();
			#ifdef _WIN32
			WaitForSingleObject(fb.locks[i], INFINITE);
			#else
			while(pthread_mutex_lock(&fb.locks[i]) != 0) {
				printf("main waiting\r\n");
			}
			#endif
			long tf_l = clock();

	        	world_update_device(&w1, 0);
			world_update_device(&w1, 1);

//			camera_render_image(&w1, fb.device_frames[i], fb.host_frames[i], 0, 0, 0, 1280, 0, 720);

			camera_render_image_splitscreen(&w1, &fb, i, 1280, 720, { 0, 1}, 0);
/*
			struct entity *entities = (struct entity *) &w1.data.data[w1.entities+1];

			float e5p1 = entities[5].position[1];

			if (e5p1 >= -10 && e5p1 <= 10) {
				e5p1 += (sh*0.05);
				world_move_entity_to_position(&w1, 5, {entities[5].position[0], e5p1, entities[5].position[2]});
			} else {
				if (once == 0) {
					once = 1;
					world_remove_entity(&w1, 4);
				}
				sh *= -1;
				e5p1 += (sh*0.1);
				world_move_entity_to_position(&w1, 5, {entities[5].position[0], e5p1, entities[5].position[2]});
			}
			harr = (harr+1)%360;
			world_set_entity_orientation(&w1, 5, {0.0, harr, 0.0});
*/

			struct camera *cameras = (struct camera *) &w1.data.data[w1.cameras+1];
//			world_set_camera_orientation(&w1, 0, {cameras[0].orientation[0], cameras[0].orientation[1], cameras[0].orientation[2]+1.0f});

			//51°26'44.4"N 6°47'39.1"E
			float px = cameras[0].position[0];
			float py = cameras[0].position[1];
			float pz = cameras[0].position[2];
			if (cameras[0].position[0] != 6.49) {
				px += ((15/100.0)*(6.79-cameras[0].position[0]));
			}
			if (cameras[0].position[1] != 51.44) {
				py += ((15/100.0)*(51.44-cameras[0].position[1]));
			}
			if (cameras[0].position[2] > 0.4) {
				pz += ((15/100.0)*(0.4-cameras[0].position[2]));
			}
			world_move_camera_to_position(&w1, 0, {px, py, pz});

/*
                       harr = (harr+1);
                        world_set_entity_orientation(&w1, 0, {0.0, harr, 0.0});
                       world_set_entity_orientation(&w1, 1, {harr, 0.0, 0.0});
*/

			#ifdef _WIN32
			ReleaseMutex(fb.locks[i]);
			#else
			pthread_mutex_unlock(&fb.locks[i]);
			#endif
			long tf_3 = clock();
			long tf_ = tf_3 -tf;
			sec += ((double)tf_/CLOCKS_PER_SEC);
			fps++;

			if (sec >= 1.0) {
                        	printf("main fps: %d\r\n", fps);
				printf("main ft: %f, main ft_l: %f\r\n", tf_/(double)CLOCKS_PER_SEC, (tf_3 - tf_l)/(double)CLOCKS_PER_SEC);
	                        sec = 0;
        	                fps = 0;
                	}
        	}

	}

/*
	for (int i = 0; i < fb_len; i++) {
		char buf[123];
		sprintf(buf, "kernel_1_fb_%i.png", i);
		lodepng::encode(&buf[0], fb.host_frames[i], c1.resolution[0], c1.resolution[1]);
		sprintf(buf, "kernel_2_fb_%i.png", i);
	        lodepng::encode(&buf[0], &fb.host_frames[i][c1.resolution[0]*c1.resolution[1]*4], c2.resolution[0], c2.resolution[1]);
	}
*/
	return 0;
}
