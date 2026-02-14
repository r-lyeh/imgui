make -DSUPPORT_WGSL=1 simple_wgsl\*.c wgvk\src\simple_wgsl_c_api.c -I simple_wgsl\_deps\spirv_headers-src\include -I simple_wgsl /Zi %* 
