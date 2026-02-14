if "%1"=="" del *.obj *.exe *.pdb *.ilk & exit /b
cl basic_wgsl_shader.c -I ..\..\glfw3\include -I ..\include ..\src\wgvk.c /experimental:c11atomics /std:c11 /nologo ..\..\glfw3\glob.c -D_GLFW_WIN32 user32.lib gdi32.lib shell32.lib -DSUPPORT_WGSL=1 ..\src\simple_wgsl_c_api.c -I ..\..\simple_wgsl\_deps\spirv_headers-src\include -I ..\..\simple_wgsl ..\..\simple_wgsl\*.c* 
