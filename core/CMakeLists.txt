file(GLOB_RECURSE core_header "*.h")
file(GLOB_RECURSE core_source "*.cpp")

if(UNIX)
    find_package(GMP REQUIRED)
    Message(STATUS "GMP_INCLUDE_DIR: ${GMP_INCLUDE_DIR}")
endif()

# Create a shared library
add_library(core_lib SHARED ${core_source})

# Include headers in build process
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${PCL_INCLUDE_DIRS})

if(UNIX)
    # Link libraries
    target_link_libraries(core_lib 
        utility_lib
        ${PCL_LIBRARIES}
        geometry_3d_lib
        geometry_2d_lib
        gmp
        )
else()
    # Link libraries
    target_link_libraries(core_lib 
        utility_lib
        ${PCL_LIBRARIES}
        geometry_3d_lib
        geometry_2d_lib
        )
endif()