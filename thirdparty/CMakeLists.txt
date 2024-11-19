file(GLOB O3D_FILES "${CMAKE_CURRENT_SOURCE_DIR}/open3d/*.cpp")

# Create a shared library
add_library(thirdparty_lib SHARED ${O3D_FILES})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/open3d/*.h")
