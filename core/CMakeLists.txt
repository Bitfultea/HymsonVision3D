file(GLOB_RECURSE core_header "*.h")
file(GLOB_RECURSE core_source "*.cpp")

# Create a shared library
add_library(core_lib SHARED ${core_source})

# Include headers in build process
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(core_lib 
    utility_lib
    ${PCL_LIBRARIES}
    geometry_3d_lib
    )