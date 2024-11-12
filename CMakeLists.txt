cmake_minimum_required( VERSION 3.10 )

project( HymsonVisionProject )

set(CMAKE_BUILD_TYPE Release) # Release,RelWithDebInfo,MinSizeRel,Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fsanitize=address") # use google sanitizer
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ")

  endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtorch")

# find the necessary packages 
find_package(PythonInterp REQUIRED) # for libtorch depenency
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(PCL REQUIRED)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


set(INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/geometry
  ${PROJECT_SOURCE_DIR}/utility
  ${PROJECT_SOURCE_DIR}/core
  # ${CMAKE_CURRENT_SOURCE_DIR}/
)

include_directories(${INCLUDE_DIRS})
include_directories(${gtest_SOURCE_DIR}/include )

# OS specific settings
if(WIN32)
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
  add_subdirectory(unitTest) 
endif()

# Utility Library
add_subdirectory(utility)

add_subdirectory(core)

# geometry
add_subdirectory(geometry)


