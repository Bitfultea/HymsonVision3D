file(GLOB O3D_FILES "C:/Users/Administrator/Desktop/hym_test/HymsonVision3D/thirdparty/open3d/*.cpp")
file(GLOB C_FILES "C:/Users/Administrator/Desktop/hym_test/HymsonVision3D/thirdparty/open3d/*.c")
add_library(thirdparty_lib SHARED ${O3D_FILES} ${C_FILES})
target_include_directories(thirdparty_lib PUBLIC
    "C:/Users/Administrator/Desktop/hym_test/HymsonVision3D/thirdparty/open3d"
    "C:/Users/Administrator/Desktop/hym_test/HymsonVision3D/thirdparty/open3d/Qhull/include"
)
#include_directories("C:/Users/zhuwenbin/Desktop/hym_test/HymsonVision3D/thirdparty/open3d/*.h")
#include_directories("C:/Users/zhuwenbin/Desktop/hym_test/HymsonVision3D/thirdparty/open3d/Qhull/include")
# Create a shared library

# target_include_directories(thirdparty_lib PUBLIC /usr/include/nanoflann)
#target_include_directories(thirdparty_lib PUBLIC "C:/Users/zhuwenbin/Desktop/hym_test/HymsonVision3D/thirdparty/open3d/Qhull/include")
#link_directories("D:/vcpkg/vcpkg/installed/x64-windows/lib")
#link_libraries(qhullcpp)

target_link_libraries(thirdparty_lib nanoflann::nanoflann fmt::fmt  Qhull::qhullcpp Qhull::qhull_r)

set_target_properties(thirdparty_lib PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})