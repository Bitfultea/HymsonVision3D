#include "Converter.h"
#include "Logger.h"

int main() {
    hymson3d::core::converter::tiff_to_pointcloud(
            "/home/charles/Data/Dataset/Collected/NG/1/NG/"
            "35596_00_06_11_876_0KECB70L000009EAB1229832_"
            "0KECB70L000009EAB1229844.tiff");
}