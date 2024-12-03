# file(GLOB_RECURSE UNITTEST_HEADER "*.h")
add_executable(LoggerTest
  # ${UNITTEST_HEADER}
  logger_test.cpp
)

add_executable(TorchTest
  test_torch.cpp
)

add_executable(TorchTest2
  test_torch_2.cpp
)

add_executable(ConvertTiff
  convert_tiff.cpp)

add_executable(
  CGAL_test
  test_CGAL.cpp
)

add_executable(
  planar_test
  test_plane.cpp
)

enable_testing()

add_executable(testGoogleTest
  google_test_sample.cpp
)

target_link_libraries(
  testGoogleTest
  gtest
  gmock
  gtest_main
  utility_lib
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

include(GoogleTest)
gtest_discover_tests(testGoogleTest)