#ifndef SDLSHOW_HPP
#define SDLSHOW_HPP

#include <SDL.h>
#include "windows.h"

void sdl_show_window();
void sdl_update_frame(void* pixels, bool capture_mouse);

#endif