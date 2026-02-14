copy /y sdl3\lib\x64\*.dll
if "%1"=="tidy" del *.pdb & del *.exe & del *.obj & del *.exp & del *.lib & del *.ini & del *.dll & exit /b
cl /nologo main.cpp -I ..\.. -I ..\..\backends -I ..\..\examples\example_glfw_wgpu\wgvk\include -I sdl3\include -DSDL_MAIN_HANDLED sdl3\lib\x64\SDL3.lib ..\..\examples\example_glfw_wgpu\wgvk\src\wgvk.c /experimental:c11atomics /std:c11 ..\..\*.c* ..\..\backends\imgui_impl_sdl3.cpp ..\..\backends\imgui_impl_wgpu.cpp -DIMGUI_IMPL_WEBGPU_BACKEND_WGVK %* 
