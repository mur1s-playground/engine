#ifdef _WIN32
#include <SDL.h>
#else
#include <SDL/SDL.h>
#endif

#include <string>
#include <iostream>
#include <time.h>

#include "SDLShow.hpp"
#include "World.hpp"

int sdl_show(const std::string& caption, const unsigned char* rgba, unsigned w, unsigned h)
{
  //avoid too large window size by downscaling large image
  unsigned jump = 1;
  if(w / 1024 >= jump) jump = w / 1024 + 1;
  if(h / 1024 >= jump) jump = h / 1024 + 1;

  //init SDL
  if(SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    std::cout << "error, SDL video init failed" << std::endl;
    return 0;
  }
  SDL_Surface* scr = SDL_SetVideoMode(w / jump, h / jump, 32, SDL_HWSURFACE);
  if(!scr)
  {
    std::cout << "error, no SDL screen" << std::endl;
    return 0;
  }
  SDL_WM_SetCaption(caption.c_str(), NULL); //set window caption

  //plot the pixels of the PNG file
  for(unsigned y = 0; y + jump - 1 < h; y += jump)
  for(unsigned x = 0; x + jump - 1 < w; x += jump)
  {
    //get RGBA components
    Uint32 r = rgba[4 * y * w + 4 * x + 0]; //red
    Uint32 g = rgba[4 * y * w + 4 * x + 1]; //green
    Uint32 b = rgba[4 * y * w + 4 * x + 2]; //blue
    Uint32 a = rgba[4 * y * w + 4 * x + 3]; //alpha

    //make translucency visible by placing checkerboard pattern behind image
    int checkerColor = 191 + 64 * (((x / 16) % 2) == ((y / 16) % 2));
    r = (a * r + (255 - a) * checkerColor) / 255;
    g = (a * g + (255 - a) * checkerColor) / 255;
    b = (a * b + (255 - a) * checkerColor) / 255;

    //give the color value to the pixel of the screenbuffer
    Uint32* bufp;
    bufp = (Uint32 *)scr->pixels + (y * scr->pitch / 4) / jump + (x / jump);
    *bufp = 65536 * r + 256 * g + b;
  }

  //pause until you press escape and meanwhile redraw screen
  SDL_Event event;
  int done = 0;
  while(done == 0)
  {
    while(SDL_PollEvent(&event))
    {
      if(event.type == SDL_QUIT) done = 2;
      else if(SDL_GetKeyState(NULL)[SDLK_ESCAPE]) done = 2;
      else if(event.type == SDL_KEYDOWN) done = 1; //press any other key for next image
    }
    SDL_UpdateRect(scr, 0, 0, 0, 0); //redraw screen
    SDL_Delay(5); //pause 5 ms so it consumes less processing power
  }

  SDL_Quit();
  return done == 2 ? 1 : 0;
}

#ifdef _WIN32
DWORD WINAPI sdl_show_loop(LPVOID param) {
#else
void *sdl_show_loop(void *param) { //const std::string& caption, struct framebuffer *fb, unsigned w, unsigned h) {
#endif
	//avoid too large window size by downscaling large image
	unsigned int w = 1280; //fb->host_worlds[0]->cameras[0].resolution[0];
        unsigned int h = 720;

	unsigned jump = 1;
//	if(w / 1024 >= jump) jump = w / 1024 + 1;
//	if(h / 1024 >= jump) jump = h / 1024 + 1;
	struct framebuffer *fb = (struct framebuffer *) param;

	//unsigned int w = 1280; //fb->host_worlds[0]->cameras[0].resolution[0];
	//unsigned int h = 720; //fb->host_worlds[0]->cameras[0].resolution[1];


	//init SDL
	if(SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cout << "error, SDL video init failed" << std::endl;
		return NULL;
	}
	SDL_Surface* scr = SDL_SetVideoMode(w / jump, h / jump, 32, SDL_HWSURFACE);
	if(!scr) {
		std::cout << "error, no SDL screen" << std::endl;
		return NULL;
	}
	SDL_WM_SetCaption("window", NULL);

	int current = 0;

	SDL_Event event;
	int done = 0;
	double sec = 0;
	int fps = 0;
	while (done == 0) {
//		long t1 = clock();

		if (current == fb->len) current = 0;
		#ifdef _WIN32
		WaitForSingleObject(fb->locks[current], INFINITE);
		#else
		while(pthread_mutex_lock(&fb->locks[current]) != 0) {
			printf("sdl waiting...\r\n");
		}
		#endif
//		long t1_l = clock();
		unsigned char *rgba = fb->host_frames[current];

		for(unsigned y = 0; y + jump - 1 < h; y += jump) {
			for(unsigned x = 0; x + jump - 1 < w; x += jump) {
			    //get RGBA components
			    Uint32 r = rgba[4 * y * w + 4 * x + 0]; //red
			    Uint32 g = rgba[4 * y * w + 4 * x + 1]; //green
			    Uint32 b = rgba[4 * y * w + 4 * x + 2]; //blue
			    Uint32 a = rgba[4 * y * w + 4 * x + 3]; //alpha

			    //make translucency visible by placing checkerboard pattern behind image
			    int checkerColor = 191 + 64 * (((x / 16) % 2) == ((y / 16) % 2));
			    r = (a * r + (255 - a) * checkerColor) / 255;
			    g = (a * g + (255 - a) * checkerColor) / 255;
			    b = (a * b + (255 - a) * checkerColor) / 255;

			    //give the color value to the pixel of the screenbuffer
			    Uint32* bufp;
			    bufp = (Uint32 *)scr->pixels + (y * scr->pitch / 4) / jump + (x / jump);
			    *bufp = 65536 * r + 256 * g + b;
		  	}
		}
		#ifdef _WIN32
		ReleaseMutex(fb->locks[current]);
		#else
		pthread_mutex_unlock(&fb->locks[current]);
		#endif
		while(SDL_PollEvent(&event)) {
		      if (event.type == SDL_QUIT) done = 2;
		      else if(SDL_GetKeyState(NULL)[SDLK_ESCAPE]) done = 2;
		      //else if(event.type == SDL_KEYDOWN) done = 1; //press any other key for next image
		}
		SDL_UpdateRect(scr, 0, 0, 0, 0);
//		SDL_Delay(5);
//		pthread_mutex_unlock(&fb->locks[current]);
/*		long t3 = clock();
		long t2 = t3 - t1;
		sec += ((double)t2/(double)CLOCKS_PER_SEC);
		fps++;
		if (sec >= 1.0) {
			printf("sdl fps: %i\r\n", fps);
			sec = 0;
			fps = 0;
		}
		if (current == 0) {
			printf("ft: %f, ft_l: %f\r\n", t2*1000.0/(CLOCKS_PER_SEC), (t3 - t1_l)/(double)CLOCKS_PER_SEC);
		}*/
		current++;

	}
	SDL_Quit();
	return NULL;
}
