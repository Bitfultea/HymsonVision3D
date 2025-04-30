file(GLOB_RECURSE pipe_header "*.h")
file(GLOB_RECURSE pipe_source "*.cpp")
# Create a shared library
add_library(pipeline SHARED ${pipe_source})
# Include headers in build process
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(pipeline 
    utility_lib
    core_lib
    #fmt::fmt
    )
if(WIN32)
    set_target_properties(pipeline PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
endif()