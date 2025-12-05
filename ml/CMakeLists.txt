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
    set(TENSORRT_ROOT "D:/software/TensorRT-8.6.1.6")
    set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
    set(TENSORRT_LIBRARY_DIR "${TENSORRT_ROOT}/lib")
    target_include_directories(ml3d_lib PUBLIC
        ${TENSORRT_INCLUDE_DIR}
    )
    target_link_directories(ml3d_lib PUBLIC
        ${TENSORRT_LIBRARY_DIR}
    )
    add_library(nvinfer STATIC IMPORTED)
    set_target_properties(nvinfer PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_LIBRARY_DIR}/nvinfer.lib"
    )

    add_library(nvinfer_plugin STATIC IMPORTED)
    set_target_properties(nvinfer_plugin PROPERTIES
        IMPORTED_LOCATION "${TENSORRT_LIBRARY_DIR}/nvinfer_plugin.lib"
    )
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
