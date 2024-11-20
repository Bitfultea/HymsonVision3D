file(GLOB O3D_FILES "${CMAKE_CURRENT_SOURCE_DIR}/open3d/*.cpp")

find_package(nanoflann REQUIRED)

# Create a shared library
add_library(thirdparty_lib SHARED ${O3D_FILES})
# target_include_directories(thirdparty_lib PUBLIC /usr/include/nanoflann)

target_link_libraries(thirdparty_lib nanoflann::nanoflann)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/open3d/*.h")

