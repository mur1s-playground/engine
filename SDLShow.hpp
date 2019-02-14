#ifndef SDLSHOW_H
#define SDLSHOW_H

#include "Framebuffer.hpp"

int sdl_show(const std::string& caption, const unsigned char* rgba, unsigned w, unsigned h);

#ifdef _WIN32
DWORD WINAPI sdl_show_loop(LPVOID lpParam);
#else
void *sdl_show_loop(void *param);	/* struct framebuffer */
#endif

#endif /* SDLSHOW_H */
