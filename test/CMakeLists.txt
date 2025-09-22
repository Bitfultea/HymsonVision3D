# file(GLOB_RECURSE UNITTEST_HEADER "*.h")
add_executable(LoggerTest
  # ${UNITTEST_HEADER}
  logger_test.cpp
)
if(WIN32)
set_target_properties(LoggerTest PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
endif()

add_executable(TorchTest
  test_torch.cpp
)
##########################################################################
if(WIN32)
  set_target_properties(TorchTest PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
  target_compile_options(TorchTest PRIVATE "/utf-8")
endif()

add_executable(TorchTest2
  test_torch_2.cpp
)

if(WIN32)
  set_target_properties(TorchTest2 PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
  target_compile_options(TorchTest2 PRIVATE "/utf-8")
endif()
##########################################################################

add_executable(ConvertTiff
  convert_tiff.cpp)

if(WIN32)
  set_target_properties(ConvertTiff PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
endif()
add_executable(
  CGAL_test
  test_CGAL.cpp
)

##########################################################
if(WIN32)
set_target_properties(CGAL_test PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
target_compile_options(CGAL_test PRIVATE "/utf-8")
endif()

add_executable(
  planar_test
  test_plane.cpp
)
if(WIN32)
set_target_properties(planar_test PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
endif()

add_executable(
  test_plana_2
  test_plane_2.cpp
)
if(WIN32)
set_target_properties(test_plana_2 PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
endif()

#########################################################
add_executable(
  def_test
  test_defect.cpp
)
if(WIN32)
set_target_properties(def_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${exe_path})
endif()

##########################################################
add_executable(
  calib_test
  test_calib.cpp
)
if(WIN32)
set_target_properties(calib_test PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
endif()


##########################################################
add_executable(
  bspline_test
  test_bspline.cpp
)
if(WIN32)
set_target_properties(bspline_test PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
endif()

##########################################################

enable_testing()





target_link_libraries(
  test_plana_2
  utility_lib
  core_lib
  geometry_3d_lib
  pipeline
  core_lib
)

target_link_libraries(
  planar_test
  utility_lib
  core_lib
  geometry_3d_lib
  pipeline
  core_lib
)

target_link_libraries(
  calib_test
  utility_lib
  core_lib
  geometry_3d_lib
  pipeline
)

target_link_libraries(
  def_test
  utility_lib
  core_lib
  geometry_3d_lib
  pipeline
  core_lib
)


target_link_libraries(
  bspline_test
  utility_lib
  core_lib
  geometry_3d_lib
  pipeline
  core_lib
)


target_link_libraries(
  LoggerTest
  utility_lib
)

target_link_libraries(
  TorchTest
  "${TORCH_LIBRARIES}"
)

target_link_libraries(
  TorchTest2
  "${TORCH_LIBRARIES}"
)

target_link_libraries(
  ConvertTiff
  core_lib
  geometry_3d_lib
  utility_lib
  thirdparty_lib
  )

