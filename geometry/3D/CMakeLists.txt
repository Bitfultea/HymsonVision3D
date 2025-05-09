file(GLOB_RECURSE 3D_source "*.cpp")
file(GLOB_RECURSE 3D_header "*.h")

# Create a shared library
add_library(geometry_3d_lib SHARED ${3D_source})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Link libraries
if(WIN32)
    target_link_libraries(geometry_3d_lib utility_lib thirdparty_lib CGAL::CGAL )
    set_target_properties(geometry_3d_lib PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
else()
    target_link_libraries(geometry_3d_lib utility_lib thirdparty_lib )
endif()
