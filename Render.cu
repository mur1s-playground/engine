
#include <stdio.h>
#include <cuda_runtime.h>

#include "Render.hpp"

__host__ __device__ vector3<float> rotate_x_ex(const vector3<float> v, const float degree) {
	float rad = degree*M_PI/180.0;
        struct matrix3<float> rot_x = { {1, 0, 0}, {0, cosf(rad), -sinf(rad)}, {0, sinf(rad), cosf(rad)}};
        return (rot_x * v);
}

__host__ __device__ vector3<float> rotate_y_ex(const vector3<float> v, const float degree) {
	float rad = degree*M_PI/180.0;
        struct matrix3<float> rot_y = { {cosf(rad), 0, sinf(rad)}, {0, 1, 0}, {-sinf(rad), 0, cosf(rad)}};
        return (rot_y * v);
}

__host__ __device__ vector3<float> rotate_z_ex(const vector3<float> v, const float degree) {
	float rad = degree*M_PI/180.0;
        struct matrix3<float> rot_z = { {cosf(rad), -sinf(rad), 0}, {sinf(rad), cosf(rad), 0}, {0, 0, 1}};
        return (rot_z * v);
}

__host__ __device__ vector3<float> rotate_x(const vector3<float> v, const float degree, const float *sin_f) {
	int rad = ((int) degree)%360;
        if (rad < 0) rad+= 360;
        if (rad >= 360) rad-= 360;
	int rad_c = (((int) degree)+90)%360;
        if (rad_c < 0) rad_c+= 360;
        if (rad_c >= 360) rad_c-= 360;
	struct matrix3<float> rot_x = { {1, 0, 0}, {0, sin_f[rad_c], -sin_f[rad]}, {0, sin_f[rad], sin_f[rad_c]}};
	return (rot_x * v);
}

__host__ __device__ vector3<float> rotate_y(const vector3<float> v, const float degree, const float *sin_f) {
        int rad = ((int) degree)%360;
        if (rad < 0) rad+= 360;
        if (rad >= 360) rad-= 360;
        int rad_c = (((int) degree)+90)%360;
        if (rad_c < 0) rad_c+= 360;
        if (rad_c >= 360) rad_c-= 360;
	struct matrix3<float> rot_y = { {sin_f[rad_c], 0, sin_f[rad]}, {0, 1, 0}, {-sin_f[rad], 0, sin_f[rad_c]}};
	return (rot_y * v);
}

__host__ __device__ vector3<float> rotate_z(const vector3<float> v, const float degree, const float *sin_f) {
        int rad = ((int) degree)%360;
        if (rad < 0) rad+= 360;
        if (rad >= 360) rad-= 360;
        int rad_c = (((int) degree)+90)%360;
        if (rad_c < 0) rad_c+= 360;
        if (rad_c >= 360) rad_c-= 360;
	struct matrix3<float> rot_z = { {sin_f[rad_c], -sin_f[rad], 0}, {sin_f[rad], sin_f[rad_c], 0}, {0, 0, 1}};
	return (rot_z * v);
}

__global__ void calcPixel(	unsigned char *image, const unsigned int cam_pos, const unsigned int ent_pos, const unsigned int sin_d, const unsigned int *bf_data, const unsigned int *bf_triangles, const unsigned int *bf_textures, 
				const unsigned int dg_pos, const unsigned int *bf_dg, const unsigned int camera_id, const int phi_start, const int phi_end, const int theta_start, const int theta_end
			) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	float *sin_f = (float *) &bf_data[sin_d+1];

	vector2<int> range_x = {(int) bf_dg[dg_pos+2], (int)bf_dg[dg_pos+3]};
        float range_x_scale = *((float *) &bf_dg[dg_pos+4]);
	vector2<int> range_y = {(int) bf_dg[dg_pos+5], (int)bf_dg[dg_pos+6]};
        float range_y_scale = *((float *) &bf_dg[dg_pos+7]);
	vector2<int> range_z = {(int) bf_dg[dg_pos+8], (int)bf_dg[dg_pos+9]};
        float range_z_scale = *((float *) &bf_dg[dg_pos+10]);

	struct camera *cam = ((struct camera *) &bf_data[cam_pos+1])+camera_id;
	struct entity *entities = (struct entity *) &bf_data[ent_pos+1];
	
	vector3<float> orientation = cam->orientation;
	vector2<int> resolution = cam->resolution;
	vector2<float> fov = cam->fov;

	int x_range = phi_end - phi_start;
	int y_range = theta_end - theta_start;

	int x = (i % x_range);
	int y = (i / x_range);

//	int img_idx = ((theta_start+y)*resolution[0]*4)+((phi_start+x)*4);
	int img_idx = ((y*x_range*4)+(x*4));

	if (i < (x_range * y_range)) {

	vector3<float> dir;

        float phi = -fov[0]/2.0 + ((phi_start + x) * (fov[0]/(float)resolution[0]));
        float theta = -fov[1]/2.0 + ((theta_start + y) * (fov[1]/(float)resolution[1]));
		
		#ifdef _WIN32
		float M_PI = 3.14159265358979323846;
		#endif

	dir[0] = -(sinf(theta*M_PI/(2.0*90.0) + M_PI/2.0) * cosf(phi*M_PI/(2.0*90) + M_PI/2.0));
        dir[1] = (sinf(theta*M_PI/(2.0*90.0) + M_PI/2.0) * sinf(phi*M_PI/(2.0*90) + M_PI/2.0));
        dir[2] = (cosf(theta*M_PI/(2.0*90.0) + M_PI/2.0));
        dir = rotate_x(dir, orientation[0], sin_f);
        dir = rotate_y(dir, orientation[1], sin_f);
        dir = rotate_z(dir, orientation[2], sin_f);

	//
	//TODO: The entity grid walkthrough is super hacky and needs improvement
	//

	vector3<float> eg_grid_pos = cam->position;
	int eg_idx_x = (int)((cam->position[0]) - range_x[0])/range_x_scale;
        int eg_idx_y = (int)((cam->position[1]) - range_y[0])/range_y_scale;
        int eg_idx_z = (int)((cam->position[2]) - range_z[0])/range_z_scale;

	unsigned int eg_last_idx = 0;
	//maybe try something with adaptive stepsize depending on grid scales and direction
	float eg_stepsize = 0.25;

	unsigned int range_fac_x = (range_x[1]-range_x[0])/range_x_scale;
	unsigned int range_fac_y = (range_y[1]-range_y[0])/range_y_scale;
	unsigned int range_fac_z = (range_z[1]-range_z[0])/range_z_scale;
	unsigned int eg_current_idx = ((eg_idx_x)*(range_fac_y*range_fac_z))+((eg_idx_y)*(range_fac_z))+eg_idx_z;

	int broke = 0;

	while (eg_idx_x >= 0 && eg_idx_x < range_fac_x && eg_idx_y >= 0 && eg_idx_y < range_fac_y && eg_idx_z >= 0 && eg_idx_z < range_fac_z) {
		while (eg_current_idx == eg_last_idx) {
			eg_grid_pos = eg_grid_pos - (-dir*eg_stepsize);
			eg_idx_x = (int)((eg_grid_pos[0]) - range_x[0])/range_x_scale;
			if (eg_idx_x < 0 || eg_idx_x >= range_fac_x) { broke = 1; break; }
		        eg_idx_y = (int)((eg_grid_pos[1]) - range_y[0])/range_y_scale;
			if (eg_idx_y < 0 || eg_idx_y >= range_fac_y) { broke = 1; break; }
		        eg_idx_z = (int)((eg_grid_pos[2]) - range_z[0])/range_z_scale;
			if (eg_idx_z < 0 || eg_idx_z >= range_fac_z) { broke = 1; break; }
			eg_current_idx = ((eg_idx_x)*(range_fac_y*range_fac_z))+((eg_idx_y)*(range_fac_z))+eg_idx_z;
		}
		if (broke) break;

		eg_last_idx = eg_current_idx;

		if (eg_current_idx >= bf_dg[dg_pos+1]) { broke = 1; break; }

		unsigned int eg_current_val = bf_dg[dg_pos+11+eg_current_idx];
		if (eg_current_val == 0) continue;
		const unsigned int *entity_indices = &bf_dg[eg_current_val];
		unsigned int entities_c = entity_indices[0];
	        float closest = -1.0;
		int closest_id_entity = -1;
		int closest_id_triangle = -1;
		int closest_id_texture = -1;

		for (int j = 0; j < entities_c; j++) {
			struct entity tmp_entity = entities[entity_indices[j+1]];
			vector3<float> position = tmp_entity.position;
			float *triangles = (float *) &bf_triangles[tmp_entity.triangles+1];
			float radius = tmp_entity.radius;

			vector3<float> r_proj = cam->position - -(dir*(dot(position-cam->position, dir)/dot(dir, dir)));
			float proj_dist = sqrtf(dot(r_proj-position, r_proj-position));
			if (proj_dist > radius) {
				continue;
			}

			float back_lambda = sqrtf((radius*radius)-(proj_dist*proj_dist));
			vector3<float> r_intersect = r_proj - (dir*back_lambda);

			unsigned int triangle_grid_pos = tmp_entity.triangle_grid;
		        vector2<int> tr_range_x = {(int) bf_triangles[triangle_grid_pos+2], (int)bf_triangles[triangle_grid_pos+3]};
			float tr_range_x_scale = *((float *) &bf_triangles[triangle_grid_pos+4]);
			vector2<int> tr_range_y = {(int) bf_triangles[triangle_grid_pos+5], (int)bf_triangles[triangle_grid_pos+6]};
			float tr_range_y_scale = *((float *) &bf_triangles[triangle_grid_pos+7]);
			vector2<int> tr_range_z = {(int) bf_triangles[triangle_grid_pos+8], (int)bf_triangles[triangle_grid_pos+9]};
			float tr_range_z_scale = *((float *) &bf_triangles[triangle_grid_pos+10]);

			float tg_stepsize = 0.1;

			unsigned int tg_last_idx = 1000000;

			//TODO: recheck the rotation stuff
			vector3<float> tg_grid_pos		= r_intersect-position;
			vector3<float> tg_grid_dir              = dir;

			tg_grid_pos = rotate_x(tg_grid_pos, tmp_entity.orientation[0], sin_f);
			tg_grid_dir = rotate_x(tg_grid_dir, tmp_entity.orientation[0], sin_f);

			tg_grid_pos = rotate_y(tg_grid_pos, tmp_entity.orientation[1], sin_f);
			tg_grid_dir = rotate_y(tg_grid_dir, tmp_entity.orientation[1], sin_f);

			tg_grid_pos = rotate_z(tg_grid_pos, tmp_entity.orientation[2], sin_f);
			tg_grid_dir = rotate_z(tg_grid_dir, tmp_entity.orientation[2], sin_f);

			int tg_idx_x = (int)(tg_grid_pos[0] - tr_range_x[0])/tr_range_x_scale;
		        int tg_idx_y = (int)(tg_grid_pos[1] - tr_range_y[0])/tr_range_y_scale;
		        int tg_idx_z = (int)(tg_grid_pos[2] - tr_range_z[0])/tr_range_z_scale;

			unsigned int tg_range_fac_x = (tr_range_x[1]-tr_range_x[0])/tr_range_x_scale;
		        unsigned int tg_range_fac_y = (tr_range_y[1]-tr_range_y[0])/tr_range_y_scale;
		        unsigned int tg_range_fac_z = (tr_range_z[1]-tr_range_z[0])/tr_range_z_scale;
		        unsigned int tg_current_idx = ((tg_idx_x)*(tg_range_fac_y*tg_range_fac_z))+((tg_idx_y)*(tg_range_fac_z))+tg_idx_z;

			int tg_broke = 0;

			while (tg_idx_x >= 0 && tg_idx_x < tg_range_fac_x && tg_idx_y >= 0 && tg_idx_y < tg_range_fac_y && tg_idx_z >= 0 && tg_idx_z < tg_range_fac_z) {
				while (tg_current_idx == tg_last_idx) {
		                        tg_grid_pos = tg_grid_pos - (-tg_grid_dir*tg_stepsize);
					tg_idx_x = (int)((tg_grid_pos[0]) - tr_range_x[0])/tr_range_x_scale;
					if (tg_idx_x < 0 || tg_idx_x >= tg_range_fac_x) { tg_broke = 1; break; }
					tg_idx_y = (int)((tg_grid_pos[1]) - tr_range_y[0])/tr_range_y_scale;
					if (tg_idx_y < 0 || tg_idx_y >= tg_range_fac_y) { tg_broke = 1; break; }
					tg_idx_z = (int)((tg_grid_pos[2]) - tr_range_z[0])/tr_range_z_scale;
					if (tg_idx_z < 0 || tg_idx_z >= tg_range_fac_z) { tg_broke = 1; break; }
		                        tg_current_idx = ((tg_idx_x)*(tg_range_fac_y*tg_range_fac_z))+((tg_idx_y)*(tg_range_fac_z))+tg_idx_z;
				}
				if (tg_broke) break;

				tg_last_idx = tg_current_idx;

		                if (tg_current_idx >= bf_triangles[triangle_grid_pos+1]) { tg_broke = 1; break; }

				unsigned int tg_current_val = bf_triangles[triangle_grid_pos+11+tg_current_idx];
		                if (tg_current_val == 0) continue;
				const unsigned int *triangle_indices = &bf_triangles[tg_current_val];
		                unsigned int triangles_c = triangle_indices[0];

			for (int t = 0; t < triangles_c; t++) {
				vector3<float> u, v, n, from, w0, w;
				float r, a, b;
				vector3<float> tr_0 = {triangles[triangle_indices[t+1]], triangles[triangle_indices[t+1]+1], triangles[triangle_indices[t+1]+2]};
				vector3<float> tr_1 = {triangles[triangle_indices[t+1]+3], triangles[triangle_indices[t+1]+4], triangles[triangle_indices[t+1]+5]};
				vector3<float> tr_2 = {triangles[triangle_indices[t+1]+6], triangles[triangle_indices[t+1]+7], triangles[triangle_indices[t+1]+8]};

				u = tr_1 - tr_0;
				v = tr_2 - tr_0;
				n = cross(u, v);

				from = tg_grid_pos - (-position);
//				from = cam->position;

				w0 = from - position - tr_0;
//				w0 = from - tr_0;
				
				a = -1 * dot(n, w0);
				b = dot(n, tg_grid_dir);
//				b = dot(n, dir);
				if (fabs(b) < 0.000000001) {
					if (a == 0) {
						continue;
					} else {
						continue;
					}
				}
				r = a / b;
				if (r < 0.0) {	
					continue;
				}
				
//				vector3<float> Il(from[0] + r * dir[0], from[1] + r * dir[1], from[2] + r * dir[2]);
				vector3<float> Il(from[0] + r * tg_grid_dir[0], from[1] + r * tg_grid_dir[1], from[2] + r * tg_grid_dir[2]);
				
				float uu, uv, vv, wu, wv, D;
				
				uu = dot(u,u);
				uv = dot(u,v);
				vv = dot(v,v);
				w = Il - position - tr_0;
				wu = dot(w,u);
				wv = dot(w,v);
				D = uv * uv - uu * vv;
				float s, ts;
				s = (uv * wv - vv * wu) / D;
				if (s < 0.0 || s > 1.0) {
					continue;
				}
				ts = (uv * wu - uu * wv) / D;
				if (ts < 0.0 || (s + ts) > 1.0) {
					continue;
				}
				if (closest < 0.0 || r < closest) {
					closest = r;
					closest_id_entity = j;
					closest_id_triangle = t;
					closest_id_texture = tmp_entity.texture_id;
				}
			}

			if (closest >= 0.0) {
				break;
			}

			} //end of tg while

		}
		if (closest >= 0.0) {	//TODO: IMPLEMENT with proper mapping
			const unsigned int *tex_start = &bf_textures[closest_id_texture+3];
			int tex_width = (int) bf_textures[closest_id_texture+1];
			int tex_height = (int) bf_textures[closest_id_texture+2];
			int val_from_x = x % tex_width;
			int val_from_y = y % tex_height;
			unsigned char *base_ptr = (unsigned char *) tex_start;
			base_ptr += ((val_from_y*tex_width*4)+(val_from_x*4));

			image[img_idx] = *base_ptr++;
			image[img_idx+1] = *base_ptr++;
			image[img_idx+2] = *base_ptr++;
			image[img_idx+3] = *base_ptr;
		} else {
			image[img_idx] = 0;
			image[img_idx+1] = 0;
			image[img_idx+2] = 0;
			image[img_idx+3] = 0;
		}		

		if (closest >= 0.0) break;
	}

	if (broke == 1) {
		image[img_idx] = 0;
                image[img_idx+1] = 0;
                image[img_idx+2] = 0;
                image[img_idx+3] = 0;
	}

	}
}

void launch_calc_pixel(struct world *w, unsigned char *d_image, unsigned int device_id, unsigned int camera_id, unsigned int phi_start, unsigned int phi_end, unsigned int theta_start, unsigned int theta_end) {
	cudaError_t err = cudaSuccess;

	long total_size = (phi_end-phi_start)*(theta_end-theta_start);
	
//	long total_size = (cam->resolution[0]*cam->resolution[1]);
	
	int threadsPerBlock = 256;
	int blocksPerGrid = (total_size + threadsPerBlock - 1) / threadsPerBlock;
//	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

	err = cudaSetDevice(device_id);	
	calcPixel<<<blocksPerGrid, threadsPerBlock>>>(d_image, w->cameras, w->entities, w->sin_d, w->data.device_data[device_id], w->triangles.device_data[device_id], w->tm->textures.device_data[device_id], w->eg.grid, w->eg.data.device_data[device_id], camera_id, phi_start, phi_end, theta_start, theta_end);
	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch calcPixel kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	//cudaDeviceSynchronize();
}
