if(WIN32)
    #set(WIN_DIR "F:/qhchen")
    file(GLOB O3D_FILES "D:/vcCode/PointCloud/thirdparty/open3d/*.cpp")
    file(GLOB C_FILES "D:/vcCode/PointCloud/thirdparty/open3d/*.c")
    add_library(thirdparty_lib SHARED ${O3D_FILES} ${C_FILES})
    target_include_directories(thirdparty_lib PUBLIC
        "D:/vcCode/PointCloud/thirdparty/open3d"
        "D:/vcCode/PointCloud/open3d/Qhull/include"
        "D:/mySoftware/vcpkg/installed/x64-windows/include/eigen3"
    )
    target_compile_definitions(thirdparty_lib PRIVATE FMT_HEADER_ONLY)
    if(MSVC)
        # MSVC 取消 FMT_SHARED（注意必须连写成一个参数）
        target_compile_options(thirdparty_lib PRIVATE "/UFMT_SHARED")
    endif()
    target_link_libraries(thirdparty_lib 
                        nanoflann::nanoflann 
                        # fmt::fmt 
                        fmt::fmt-header-only 
                        Qhull::qhullcpp 
                        Qhull::qhull_r
                        Eigen3::Eigen)
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