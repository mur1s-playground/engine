#pragma once

void render_launch(
	const unsigned int* bf_dynamic, const unsigned int cameras_position, const unsigned int camera_c, const unsigned int camera_id, const unsigned int entity_grid_position, const unsigned int entities_dynamic_position,
	const unsigned int* bf_static, const unsigned int entities_static_position, const unsigned int entities_static_count,
	const unsigned int triangles_static_position, const unsigned int triangles_static_grid_position,
	const unsigned int textures_map_static_position, const unsigned int textures_static_position);

void render_compose_kernel_launch(const unsigned char* src, const int src_width, const int src_height, const int src_channels, const int dx, const int dy, const int crop_x1, const int crop_x2, const int crop_y1, const int crop_y2, const int width, const int height, unsigned char* dst, const int dst_width, const int dst_height, const int dst_channels);

void render_staircase_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const float amplify);
void render_max_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int k);
void render_edge_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const float amplify);
void render_gauss_blur_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels);
void render_anti_alias_kernel_launch(const unsigned char* src, const unsigned char* src_s, unsigned char* dst, const int width, const int height, const int channels);