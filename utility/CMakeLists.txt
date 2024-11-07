file(GLOB_RECURSE utility_header "*.h")
file(GLOB_RECURSE utility_source "*.cpp")

# Create a shared library
add_library(utility_lib SHARED ${utility_source})

# Include headers in build process
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(utility_lib spdlog::spdlog fmt::fmt stdc++fs)