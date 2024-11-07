#pragma once

#include <functional>
#include <string>
#include <vector>

namespace hymson3d {
namespace utility {
namespace filesystem {

int TestGoogleTest(int params);

std::string GetHomeDirectory();

std::string GetFileExtensionInLowerCase(const std::string &filename);

std::string GetFileNameWithoutExtension(const std::string &filename);

std::string GetFileNameWithoutDirectory(const std::string &filename);

std::string GetFileParentDirectory(const std::string &filename);

std::string GetRegularizedDirectoryName(const std::string &directory);

std::string GetWorkingDirectory();

std::vector<std::string> GetPathComponents(const std::string &path);

std::string GetTempDirectoryPath();

bool ChangeWorkingDirectory(const std::string &directory);

bool DirectoryExists(const std::string &directory);

// Return true if the directory is present and empty. Return false if the
// directory is present but not empty. Throw an exception if the directory is
// not present.
bool DirectoryIsEmpty(const std::string &directory);

bool MakeDirectory(const std::string &directory);

bool MakeDirectoryHierarchy(const std::string &directory);

bool DeleteDirectory(const std::string &directory);

bool FileExists(const std::string &filename);

bool Copy(const std::string &src_path, const std::string &dst_path);

bool RemoveFile(const std::string &filename);

bool ListDirectory(const std::string &directory,
                   std::vector<std::string> &subdirs,
                   std::vector<std::string> &filenames);

bool ListFilesInDirectory(const std::string &directory,
                          std::vector<std::string> &filenames);

bool ListFilesInDirectoryWithExtension(const std::string &directory,
                                       const std::string &extname,
                                       std::vector<std::string> &filenames);

std::vector<std::string> FindFilesRecursively(
        const std::string &directory,
        std::function<bool(const std::string &)> is_match);

// wrapper for fopen that enables unicode paths on Windows
FILE *FOpen(const std::string &filename, const std::string &mode);
std::string GetIOErrorString(const int errnoVal);
bool FReadToBuffer(const std::string &path,
                   std::vector<char> &bytes,
                   std::string *errorStr);

std::string JoinPath(const std::vector<std::string> &path_components);

std::string JoinPath(const std::string &path_component1,
                     const std::string &path_component2);

std::string AddIfExist(const std::string &path,
                       const std::vector<std::string> &folder_names);
}  // namespace filesystem
}  // namespace utility
}  // namespace hymson3d