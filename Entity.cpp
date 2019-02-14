#include "Entity.hpp"

void entity_init(struct entity *e, unsigned int triangles, float radius) {
	e->position = {0.0, 0.0, 0.0};
	e->orientation = { 0.0, 0.0, 0.0 };
        e->scale = { 1.0, 1.0, 1.0 };

        e->triangles = triangles;
        e->radius = radius;
	e->triangle_grid = 0;

	e->texture_id = 0;
}

struct entity *entity_generate_cube(unsigned int triangles) {
        struct entity *cube = (struct entity *) malloc(sizeof(struct entity));
        cube->position = { 0.0, 0.0, 0.0 };
        cube->orientation = { 0.0, 0.0, 0.0 };
        cube->scale = { 1.0, 1.0, 1.0 };

        cube->triangles = triangles;
        cube->radius = sqrt(3)/2.0f;
	cube->triangle_grid = 0;

	cube->texture_id = 0;
	return cube;
}

float *entity_generate_cube_triangles(unsigned int *out_len) {
	int size = 36*3;
	float *vecs = (float *) malloc(size*sizeof(float));
	*out_len = size;

	/* LEGACY */
	struct vector3<float> *v = (struct vector3<float> *) malloc(36*sizeof(struct vector3<float>));

	//bottom 1
	v[0] = { -0.5, -0.5, -0.5 };
	v[1] = {  0.5, -0.5, -0.5 };
	v[2] = {  0.5,  0.5, -0.5 };
	//bottom 2
	v[3] = { -0.5, -0.5, -0.5 };
	v[4] = {  0.5,  0.5, -0.5 };
	v[5] = { -0.5,  0.5, -0.5 };
	//top 1
	v[6] = { -0.5, -0.5,  0.5 };
	v[7] = {  0.5, -0.5,  0.5 };
	v[8] = {  0.5,  0.5,  0.5 };
	//top 2
	v[9] =  { -0.5, -0.5,  0.5 };
	v[10] = {  0.5,  0.5,  0.5 };
	v[11] = { -0.5,  0.5,  0.5 };
	//front 1
	v[12] = { -0.5, -0.5, -0.5 };
	v[13] = {  0.5, -0.5, -0.5 };
	v[14] = {  0.5, -0.5,  0.5 };
	//front 2
	v[15] = { -0.5, -0.5, -0.5 };
	v[16] = {  0.5, -0.5,  0.5 };
	v[17] = { -0.5, -0.5,  0.5 };
	//back 1
	v[18] = { -0.5,  0.5, -0.5 };
	v[19] = {  0.5,  0.5, -0.5 };
	v[20] = {  0.5,  0.5,  0.5 };
	//back 2
	v[21] = { -0.5,  0.5, -0.5 };
	v[22] = {  0.5,  0.5,  0.5 };
	v[23] = { -0.5,  0.5,  0.5 };
	//left 1
	v[24] = { -0.5, -0.5, -0.5 };
	v[25] = { -0.5, -0.5,  0.5 };
	v[26] = { -0.5,  0.5, -0.5 };
	//left 2
	v[27] = { -0.5,  0.5, -0.5 };
	v[28] = { -0.5, -0.5,  0.5 };
	v[29] = { -0.5,  0.5,  0.5 };
	//right 1
	v[30] = {  0.5, -0.5, -0.5 };
	v[31] = {  0.5, -0.5,  0.5 };
	v[32] = {  0.5,  0.5, -0.5 };
	//right 2
	v[33] = {  0.5,  0.5, -0.5 };
	v[34] = {  0.5, -0.5,  0.5 };
	v[35] = {  0.5,  0.5,  0.5 };

	for (int i = 0; i < 36; i++) {
		for (int j = 0; j < 3; j++) {
			vecs[(i*3)+j] = v[i][j];
		}
	}

	return vecs;
}

unsigned char *entity_generate_default_texture(int tw, int th, int t, int r, int g, int b) {
	unsigned char *tex = (unsigned char *) malloc(4*tw*th*sizeof(unsigned char));
	for (int i = 0; i < tw; i++) {
			for (int j = 0; j < th; j++) {
				if (j < th/2) {
					tex[(i*th*4)+(j*4)] = r;
					tex[(i*th*4)+(j*4)+1] = g;
					tex[(i*th*4)+(j*4)+2] = b;
					tex[(i*th*4)+(j*4)+3] = 255;
				} else {
					tex[(i*th*4)+(j*4)] = r;
					tex[(i*th*4)+(j*4)+1] = g;
					tex[(i*th*4)+(j*4)+2] = b;
					tex[(i*th*4)+(j*4)+3] = 255;
				}
			}
	}
	return tex;
}
