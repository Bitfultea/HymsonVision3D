cmake_minimum_required( VERSION 3.10 )
# ���� Windows ƽ̨ȫ�ֽ��� fmt �ı����ڼ��
if (WIN32)
  set(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/vcpkg-master/scripts/buildsystems/vcpkg.cmake")
  add_compile_definitions(FMT_ENFORCE_COMPILE_CHECK=0)
endif()
project( HymsonVisionProject )

set(CMAKE_BUILD_TYPE Release) # Release,RelWithDebInfo,MinSizeRel,Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fsanitize=address") # use google sanitizer
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ")
endif()

if(NOT WIN32)
  set(CMAKE_CUDA_ARCHITECTURES "86") # rtx 3090
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
else ()
  set(vcpkg_root_dir "D:/vcpkg/vcpkg-master/packages")
  set(vcpkg_include_dir "D:/vcpkg/vcpkg-master/installed/x64-windows/include")
  set(vcpkg_lib_dir "D:/vcpkg/vcpkg-master/installed/x64-windows/lib")
  set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc")
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "D:/software/opencv-4.8.0/build")
  set(exe_path "${CMAKE_CURRENT_SOURCE_DIR}/build/exe")
endif()

set(CGAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/CGAL-5.6.2")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/CGAL-5.6.2")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtorch")

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# find the necessary packages 
find_package(PythonInterp REQUIRED) # for libtorch depenency
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Qhull REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(PCL REQUIRED)
find_package(CGAL REQUIRED)
find_package(nanoflann REQUIRED)
if(WIN32)
  find_package(boost REQUIRED)
  find_package(TIFF REQUIRED)
endif()

#find_package(Open3D REQUIRED)
#find_package(lz4 CONFIG REQUIRED)


#include(FetchContent)
#FetchContent_Declare(
  #googletest
  #URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
#)

# For Windows: Prevent overriding the parent project's compiler/linker settings
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)

if(WIN32)
  set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/geometry
    ${PROJECT_SOURCE_DIR}/utility
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/thirdparty
    ${PROJECT_SOURCE_DIR}/pipeline
    ${PROJECT_SOURCE_DIR}/ml
    "'D:/software/opencv-4.8.0/build/include"
    ${vcpkg_include_dir}
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
    ${PROJECT_SOURCE_DIR}/thirdparty/CGAL-5.6.2/auxiliary/gmp/include
  )
elseif(UNIX)
  set(INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/geometry
    ${PROJECT_SOURCE_DIR}/utility
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/thirdparty
    ${PROJECT_SOURCE_DIR}/pipeline
    ${PROJECT_SOURCE_DIR}/ml
    # /usr/include/mkl # include intel mkl for eigen hpc
  )
endif()

include_directories(${INCLUDE_DIRS})
include_directories(${gtest_SOURCE_DIR}/include )
include_directories(${CGAL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
#include_directories("${vcpkg_root_dir}/fmt_x64-windows/include")
#include_directories("${vcpkg_root_dir}/qhull_x64-windows/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/open3d")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtorch/include/torch/csrc/api/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtorch/include")
# OS specific settings
if(WIN32)
    include_directories(${TIFF_INCLUDE_DIR})
    # Windows defaults to hidden symbol visibility, override that
    # TODO: It would be better to explicitly export symbols.
    #       Then, we could use -fvisibility=hidden for Linux as well
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MSVC)
        # Make sure we don't hit the 65535 object member limit with MSVC
        #
        # /bigobj allows object files with more than 65535 members
        # /Ob2 enables function inlining, because MSVC is particularly
        # verbose with inline members
        #
        # See: https://github.com/tensorflow/tensorflow/pull/10962
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj;/Ob2>")
    endif()
    if (STATIC_WINDOWS_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

# Unit test
option(BUILD_UNIT_TESTS "Build Unit Tests" ON)
if(BUILD_UNIT_TESTS)
  add_subdirectory(test) 
endif()

if(UNIX)
  # add_definitions(-DEIGEN_USE_MKL_ALL)
endif()
add_subdirectory(utility)
add_subdirectory(thirdparty)
add_subdirectory(core)
add_subdirectory(geometry)
add_subdirectory(ml)
add_subdirectory(pipeline)

if(WIN32)
  set(INSTALL_FOLDER  "${CMAKE_CURRENT_SOURCE_DIR}/build/deploy")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/build")
  # set(INSTALL_FOLDER  "deploy")

  # # Install all library targets with their runtime dependencies
  # install(TARGETS utility_lib pipeline core_lib geometry_3d_lib geometry_2d_lib thirdparty_lib
  #         RUNTIME_DEPENDENCIES
  #         PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*"
  #         POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
  #         RUNTIME DESTINATION ${INSTALL_FOLDER}
  #         LIBRARY DESTINATION ${INSTALL_FOLDER}
  #         ARCHIVE DESTINATION ${INSTALL_FOLDER})
  
  # # Install test executables with their runtime dependencies
  # install(TARGETS LoggerTest TorchTest TorchTest2 ConvertTiff CGAL_test
  #         RUNTIME_DEPENDENCIES
  #         PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*"
  #         POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
  #         RUNTIME DESTINATION ${INSTALL_FOLDER})
endif()