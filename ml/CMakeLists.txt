add_library(ml3d_lib SHARED)

# TensorRT
# find_package(TensorRT REQUIRED)

target_sources(ml3d_lib PRIVATE
    yolo_3d/segment_3d.cpp
)

target_include_directories(ml3d_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32)
    target_link_libraries(ml3d_lib 
        spdlog::spdlog 
        fmt::fmt 
        Eigen3::Eigen
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        # ${CUDAToolkit_LIBRARIES}
        TIFF::TIFF
        nvinfer
        nvinfer_plugin
        )
    set_target_properties(ml3d_lib PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
else()
    # Link libraries
    target_link_libraries(ml3d_lib 
        spdlog::spdlog 
        fmt::fmt 
        stdc++fs 
        Eigen3::Eigen
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        nvinfer
        nvinfer_plugin
        )
endif()
