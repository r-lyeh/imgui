if "%1"=="tidy" del *.pdb *.exe *.obj *.exp *.lib *.ini *.dll *.ilk & exit /b
cl main.cpp /nologo -I ..\.. -I ..\..\backends -I wgvk\include -I glfw3\include glfw3\glob.c -D_GLFW_WIN32 shell32.lib gdi32.lib user32.lib ..\..\backends\imgui_impl_glfw.cpp ..\..\*.cpp ..\..\backends\imgui_impl_w*.cpp -DIMGUI_IMPL_WEBGPU_BACKEND_WGVK -I. wgvk\src\wgvk.c /experimental:c11atomics /std:c11 %*
