# file(GLOB_RECURSE UNITTEST_HEADER "*.h")
add_executable(LoggerTest
  # ${UNITTEST_HEADER}
  logger_test.cpp
)
set_target_properties(LoggerTest PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
add_executable(TorchTest
  test_torch.cpp
)
set_target_properties(TorchTest PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
add_executable(TorchTest2
  test_torch_2.cpp
)
set_target_properties(TorchTest2 PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
add_executable(ConvertTiff
  convert_tiff.cpp)
set_target_properties(ConvertTiff PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
add_executable(
  CGAL_test
  test_CGAL.cpp
)
set_target_properties(CGAL_test PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
add_executable(
  planar_test
  test_plane.cpp
)
set_target_properties(planar_test PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
add_executable(
  def_test
  test_defect.cpp
)
set_target_properties(def_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${exe_path})
add_executable(
  calib_test
  test_calib.cpp
)
set_target_properties(calib_test PROPERTIES 
                        RUNTIME_OUTPUT_DIRECTORY ${exe_path})
enable_testing()





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

