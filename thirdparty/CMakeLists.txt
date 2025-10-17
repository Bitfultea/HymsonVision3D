if(WIN32)
    set(WIN_DIR "F:/qhchen")
    file(GLOB O3D_FILES "${WIN_DIR}/HymsonVision3D/thirdparty/open3d/*.cpp")
    file(GLOB C_FILES "${WIN_DIR}/HymsonVision3D/thirdparty/open3d/*.c")
    add_library(thirdparty_lib SHARED ${O3D_FILES} ${C_FILES})
    target_include_directories(thirdparty_lib PUBLIC
        "${WIN_DIR}/HymsonVision3D/thirdparty/open3d"
        "${WIN_DIR}/HymsonVision3D/thirdparty/open3d/Qhull/include"
    )
    target_link_libraries(thirdparty_lib nanoflann::nanoflann fmt::fmt  Qhull::qhullcpp Qhull::qhull_r)
elseif(UNIX)
    file(GLOB O3D_FILES "${CMAKE_CURRENT_SOURCE_DIR}/open3d/*.cpp")
    file(GLOB O3D_C_FILES "${CMAKE_CURRENT_SOURCE_DIR}/open3d/*.c")
    # Create a shared library
    add_library(thirdparty_lib SHARED ${O3D_FILES} ${O3D_C_FILES})
    # target_include_directories(thirdparty_lib PUBLIC /usr/include/nanoflann)
    target_include_directories(thirdparty_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(thirdparty_lib nanoflann::nanoflann qhullcpp)

    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/open3d/*.h")
endif()


if(WIN32)
    set_target_properties(thirdparty_lib PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
endif()