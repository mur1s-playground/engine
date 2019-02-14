all: engine

engine: Main.o World.o Entity.o Camera.o TextureMapper.o Render.o external/lodepng.o Framebuffer.o SDLShow.o BitField.o EntityGrid.o Catalog.o TriangleGrid.o
	g++ Main.o World.o Entity.o Camera.o TextureMapper.o Render.o external/lodepng.o Framebuffer.o SDLShow.o BitField.o EntityGrid.o Catalog.o TriangleGrid.o -lpthread -lcudart -lSDL -o engine

Main.o: Main.cpp
	nvcc -std=c++11 -c Main.cpp -o Main.o

Catalog.o: Catalog.cpp
	nvcc -std=c++11 -c Catalog.cpp -o Catalog.o

TriangleGrid.o: TriangleGrid.cpp
	nvcc -std=c++11 -c TriangleGrid.cpp -o TriangleGrid.o

World.o: World.cpp
	nvcc -std=c++11 -c World.cpp -o World.o

Entity.o: Entity.cpp
	nvcc -std=c++11 -c Entity.cpp -o Entity.o

Camera.o: Camera.cpp
	nvcc -std=c++11 -c Camera.cpp -o Camera.o

TextureMapper.o: TextureMapper.cpp
	nvcc -std=c++11 -c TextureMapper.cpp -o TextureMapper.o

Render.o: Render.cu
	nvcc -std=c++11 -c Render.cu -o Render.o

Framebuffer.o: Framebuffer.cpp
	g++ -std=c++11 -c Framebuffer.cpp -o Framebuffer.o

EntityGrid.o: EntityGrid.cpp
	nvcc -std=c++11 -c EntityGrid.cpp -o EntityGrid.o

BitField.o: BitField.cpp
	nvcc -std=c++11 -c BitField.cpp -o BitField.o

SDLShow.o: SDLShow.cpp
	nvcc -std=c++11 -c SDLShow.cpp -o SDLShow.o

external/lodepng.o: external/lodepng.cpp
	g++ -c external/lodepng.cpp -o external/lodepng.o

clean:
	rm *.o
	rm engine
	rm external/*.o
