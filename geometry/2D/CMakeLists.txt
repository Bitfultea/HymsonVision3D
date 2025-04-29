file(GLOB_RECURSE 2D_source "*.cpp")
file(GLOB_RECURSE 2D_header "*.h")

# Create a shared library
add_library(geometry_2d_lib SHARED ${2D_source})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Link libraries
target_link_libraries(geometry_2d_lib utility_lib)
set_target_properties(geometry_2d_lib PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})