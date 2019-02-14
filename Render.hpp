#ifndef RENDER_HPP
#define RENDER_HPP

#include "cuda_runtime.h"

#include "World.hpp"
#include "Entity.hpp"
#include "Matrix3.hpp"
#include "Vector2.hpp"
#include "Vector3.hpp"
#include "TextureMapper.hpp"

__host__ __device__ vector3<float> rotate_x(const vector3<float> v, const float degree, const float *sin_f);
__host__ __device__ vector3<float> rotate_y(const vector3<float> v, const float degree, const float *sin_f);
__host__ __device__ vector3<float> rotate_z(const vector3<float> v, const float degree, const float *sin_f);

void launch_calc_pixel(struct world *w, unsigned char *d_image, unsigned int device_id, unsigned int camera_id,
unsigned int phi_start, unsigned int phi_end, unsigned int theta_start, unsigned int theta_end);

#endif
