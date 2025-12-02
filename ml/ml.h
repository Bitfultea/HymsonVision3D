#pragma once
namespace hymson3d {
namespace ml {

const std::vector<std::string> CLASS_NAMES = {"pinhole", "crap", "scatter"};
const std::vector<std::vector<unsigned int>> COLORS = {
        {0, 114, 189}, {217, 83, 25}, {237, 177, 32}};
// const std::vector<std::vector<unsigned int>> MASK_COLORS = {
//         {255, 56, 56}, {255, 157, 151}, {255, 112, 31}};
const std::vector<std::vector<unsigned int>> MASK_COLORS = {
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
}  // namespace ml
}  // namespace hymson3d