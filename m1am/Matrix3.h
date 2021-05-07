#ifndef MATRIX3_HPP
#define MATRIX3_HPP

#include <cuda_runtime.h>
#include "Vector3.h"

template<typename T>
struct matrix3 {
	vector3<T> v[3];

	__host__ __device__ matrix3() {}
	__host__ __device__ matrix3(vector3<T> v1, vector3<T> v2, vector3<T> v3) : v{ v1, v2, v3 } {}

	__host__ __device__ vector3<T>& operator[](const std::size_t n) { return v[n]; }
	__host__ __device__ const vector3<T>& operator[](const std::size_t n) const { return v[n]; }
};

template<typename T>
__host__ __device__ auto operator-(const matrix3<T> m) -> matrix3<T> {
	return { -m[0], -m[1], -m[2] };
}

template<typename T>
__host__ __device__ auto operator-(const matrix3<T> m1, const vector3<T> m2) -> matrix3<T> {
	return { m1[0] - m2[0], m1[1] - m2[1], m1[2] - m2[2] };
}

template<typename T>
__host__ __device__ auto operator*(const matrix3<T> m, const T t) -> vector3<T> {
	return { m[0] * t, m[1] * t, m[2] * t };
}

template<typename T>
__host__ __device__ auto operator*(const matrix3<T> m, const vector3<T> v) -> vector3<T> {
	return { m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2], m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2], m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] };
}

template<typename T>
__host__ __device__ auto operator*(const matrix3<T> m1, const matrix3<T> m2) -> matrix3<T> {
	vector3<T> m1_v_r0{ m1[0][0], m1[1][0], m1[2][0] };
	vector3<T> m1_v_r1{ m1[0][1], m1[1][1], m1[2][1] };
	vector3<T> m1_v_r2{ m1[0][2], m1[1][2], m1[2][2] };
	return { { dot(m1_v_r0, m2[0]), dot(m1_v_r1, m2[0]), dot(m1_v_r2, m2[0]) },
				{ dot(m1_v_r0, m2[1]), dot(m1_v_r1, m2[1]), dot(m1_v_r2, m2[1]) },
				{ dot(m1_v_r0, m2[2]), dot(m1_v_r1, m2[2]), dot(m1_v_r2, m2[2]) } };
}

template<typename T>
__host__ __device__ vector3<T> rotate_x(vector3<T> v, T rad_x) {
	struct matrix3<T> rot_x = { {1, 0, 0}, {0, cosf(rad_x), -sinf(rad_x)}, {0, sinf(rad_x), cosf(rad_x)} };
	return (rot_x * v);
}

template<typename T>
__host__ __device__ vector3<T> rotate_y(vector3<T> v, T rad_y) {
	struct matrix3<T> rot_y = { {cosf(rad_y), 0, sinf(rad_y)}, {0, 1, 0}, {-sinf(rad_y), 0, cosf(rad_y)} };
	return (rot_y * v);
}

template<typename T>
__host__ __device__ vector3<T> rotate_z(vector3<T> v, T rad_z) {
	struct matrix3<T> rot_z = { {cosf(rad_z), -sinf(rad_z), 0}, {sinf(rad_z), cosf(rad_z), 0}, {0, 0, 1} };
	return (rot_z * v);
}

#endif /* MATRIX3_HPP */
