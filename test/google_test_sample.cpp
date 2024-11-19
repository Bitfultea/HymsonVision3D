#include "FileSystem.h"
#include "gtest/gtest.h"

namespace {
TEST(googleTest, BasicAssertions) {
    EXPECT_EQ(3, hymson3d::utility::filesystem::TestGoogleTest(3)) << "Yell";
    EXPECT_EQ(5, hymson3d::utility::filesystem::TestGoogleTest(2));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    std::cout << "Running main() from google_test_sample.cc\n";
    return RUN_ALL_TESTS();
}

}  // namespace