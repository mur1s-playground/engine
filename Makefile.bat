if "%~1"=="clean" (
goto clean
) else if "%~1"=="update" (
goto update
) else (
goto all
)
:clean
del *.obj
del Main.exe

exit /B 0
:update
"C:\Users\mur1_\Downloads\pscp.exe" -r odroid@192.168.178.31:/home/odroid/Projects/engine ..\
goto setvars
:all
SET compiler="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\cl.exe"
SET cflags="/c"

SET nv_compiler="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\bin\nvcc.exe"
SET nv_flags="-c"

SET nv_libpath="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\lib\x64"
SET nv_lib_cuda="cuda.lib"
SET nv_lib_cudart="cudart.lib"

SET sdl_libpath="C:\Users\mur1_\Downloads\SDL-devel-1.2.15-VC\SDL-1.2.15\lib\x64"
SET sdl_lib="SDL.lib"

SET linker="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\link.exe"

%nv_compiler% %nv_flags% -I "./external" -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include" -I "C:\Users\mur1_\Downloads\SDL-devel-1.2.15-VC\SDL-1.2.15\include" Main.cpp World.cpp Entity.cpp Camera.cpp TextureMapper.cpp Render.cu BitField.cpp EntityGrid.cpp Framebuffer.cpp SDLShow.cpp Catalog.cpp
%compiler% %cflags% /EHsc  external/lodepng.cpp /I "C:\Users\mur1_\Downloads\SDL-devel-1.2.15-VC\SDL-1.2.15\include"
%linker% ws2_32.lib Main.obj World.obj Entity.obj Camera.obj TextureMapper.obj lodepng.obj Render.obj BitField.obj EntityGrid.obj SDLShow.obj Framebuffer.obj Catalog.obj /LIBPATH:%sdl_libpath% %sdl_lib% /LIBPATH:%nv_libpath% %nv_lib_cuda% %nv_lib_cudart%

:setvars
if defined issetvars (
SET issetvars="1"
) else (
SET issetvars="1"
"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\vcvars64.bat"
)
