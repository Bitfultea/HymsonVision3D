file(GLOB_RECURSE pipe_header "*.h")
file(GLOB_RECURSE pipe_source "*.cpp")
# Create a shared library
add_library(pipeline SHARED ${pipe_source})
# Include headers in build process
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


if(WIN32)
    # Link libraries
    target_link_libraries(pipeline 
        utility_lib
        core_lib
        ml3d_lib
        nvinfer
        spdlog::spdlog_header_only
        fmt::fmt-header-only
        # ${CUDA_LIBRARIES}
        #fmt::fmt
        )
    target_compile_definitions(pipeline PRIVATE FMT_HEADER_ONLY SPDLOG_FMT_EXTERNAL)
    if(MSVC)
        target_compile_options(pipeline PRIVATE "/UFMT_SHARED" "/USPDLOG_SHARED_LIB" "/USPDLOG_COMPILED_LIB")
    endif()
    set_target_properties(pipeline PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
else()
    # Link libraries
    target_link_libraries(pipeline 
        utility_lib
        core_lib
        ml3d_lib
        libnvinfer.so
        # ${CUDA_LIBRARIES}
        #fmt::fmt
        )

endif()