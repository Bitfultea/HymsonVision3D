file(GLOB_RECURSE utility_header "*.h")
file(GLOB_RECURSE utility_source "*.cpp")

# Create a shared library
add_library(utility_lib SHARED ${utility_source})

# Include headers in build process
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


if(WIN32)
    # Link libraries
    target_link_libraries(utility_lib 
        spdlog::spdlog 
        fmt::fmt 
        Eigen3::Eigen
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        )
    set_target_properties(utility_lib PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
else()
    # Link libraries
    target_link_libraries(utility_lib 
        spdlog::spdlog 
        fmt::fmt 
        stdc++fs 
        Eigen3::Eigen
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        )
endif()