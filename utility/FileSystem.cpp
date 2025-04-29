#include "FileSystem.h"

#include <fcntl.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#ifdef WIN32
#include <direct.h>
#include <dirent/dirent.h>
#include <io.h>
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#else
#include <dirent.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef WIN32
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#endif
#ifdef __APPLE__
#include <filesystem>
namespace fs = std::__fs::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "Logger.h"

namespace hymson3d {
namespace utility {
namespace filesystem {
int TestGoogleTest(int params) { return params; }

static std::string GetEnvVar(const std::string &env_var) {
    if (const char *env_p = std::getenv(env_var.c_str())) {
        return std::string(env_p);
    } else {
        return "";
    }
}

std::string GetHomeDirectory() {
    std::string home_dir = "";
#ifdef WINDOWS
    // %USERPROFILE%
    // %HOMEDRIVE%
    // %HOMEPATH%
    // %HOME%
    // C:/
    home_dir = GetEnvVar("USERPROFILE");
    if (home_dir.empty() || !DirectoryExists(home_dir)) {
        home_dir = GetEnvVar("HOMEDRIVE");
        if (home_dir.empty() || !DirectoryExists(home_dir)) {
            home_dir = GetEnvVar("HOMEPATH");
            if (home_dir.empty() || !DirectoryExists(home_dir)) {
                home_dir = GetEnvVar("HOME");
                if (home_dir.empty() || !DirectoryExists(home_dir)) {
                    home_dir = "C:/";
                }
            }
        }
    }
#else
    home_dir = GetEnvVar("HOME");
    if (home_dir.empty() || !DirectoryExists(home_dir)) {
        home_dir = "/";
    }
#endif
    return home_dir;
}

std::string GetFileExtensionInLowerCase(const std::string &filename) {
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos >= filename.length()) return "";

    if (filename.find_first_of("/\\", dot_pos) != std::string::npos) return "";

    std::string filename_ext = filename.substr(dot_pos + 1);

    std::transform(filename_ext.begin(), filename_ext.end(),
                   filename_ext.begin(), ::tolower);

    return filename_ext;
}

std::string GetFileNameWithoutExtension(const std::string &filename) {
    size_t dot_pos = filename.find_last_of(".");

    return filename.substr(0, dot_pos);
}

std::string GetFileNameWithoutDirectory(const std::string &filename) {
    size_t slash_pos = filename.find_last_of("/\\");
    if (slash_pos == std::string::npos) {
        return filename;
    } else {
        return filename.substr(slash_pos + 1);
    }
}

std::string GetFileParentDirectory(const std::string &filename) {
    size_t slash_pos = filename.find_last_of("/\\");
    if (slash_pos == std::string::npos) {
        return "";
    } else {
        return filename.substr(0, slash_pos + 1);
    }
}

std::string GetRegularizedDirectoryName(const std::string &directory) {
    if (directory.empty()) {
        return "/";
    } else if (directory.back() != '/' && directory.back() != '\\') {
        return directory + "/";
    } else {
        return directory;
    }
}

std::string GetWorkingDirectory() {
    char buff[PATH_MAX + 1];
    auto ignored = getcwd(buff, PATH_MAX + 1);
    (void)ignored;
    return std::string(buff);
}

std::vector<std::string> GetPathComponents(const std::string &path) {
    auto SplitByPathSeparators = [](const std::string &path) {
        std::vector<std::string> components;
        // Split path by '/' and '\'
        if (!path.empty()) {
            size_t end = 0;
            while (end < path.size()) {
                size_t start = end;
                while (end < path.size() && path[end] != '\\' &&
                       path[end] != '/') {
                    end++;
                }
                if (end > start) {
                    components.push_back(path.substr(start, end - start));
                }
                if (end < path.size()) {
                    end++;
                }
            }
        }
        return components;
    };

    auto pathComponents = SplitByPathSeparators(path.c_str());

    // Handle "/" and "" paths
    if (pathComponents.empty()) {
        if (path == "/") {
            // absolute path; the "/" component will be added
            // later, so don't do anything here
        } else {
            pathComponents.push_back(".");
        }
    }

    char firstChar = path[0];  // '/' doesn't get stored in components
    bool isRelative = (firstChar != '/');
    bool isWindowsPath = false;
    // Check for Windows full path (e.g. "d:")
    if (isRelative && pathComponents[0].size() >= 2 &&
        ((firstChar >= 'a' && firstChar <= 'z') ||
         (firstChar >= 'A' && firstChar <= 'Z')) &&
        pathComponents[0][1] == ':') {
        isRelative = false;
        isWindowsPath = true;
    }

    std::vector<std::string> components;
    if (isRelative) {
        auto cwd = utility::filesystem::GetWorkingDirectory();
        auto cwdComponents = SplitByPathSeparators(cwd);
        components.insert(components.end(), cwdComponents.begin(),
                          cwdComponents.end());
        if (cwd[0] != '/') {
            isWindowsPath = true;
        }
    } else {
        // absolute path, don't need any prefix
    }
    if (!isWindowsPath) {
        components.insert(components.begin(), "/");
    }

    for (auto &dir : pathComponents) {
        if (dir == ".") { /* ignore */
        } else if (dir == "..") {
            components.pop_back();
        } else {
            components.push_back(dir);
        }
    }

    return components;
}

std::string GetTempDirectoryPath() {
    return fs::temp_directory_path().string();
}

bool ChangeWorkingDirectory(const std::string &directory) {
    return (chdir(directory.c_str()) == 0);
}

bool DirectoryExists(const std::string &directory) {
    return fs::is_directory(directory);
}

bool DirectoryIsEmpty(const std::string &directory) {
    if (!DirectoryExists(directory)) {
        LOG_ERROR("Directory {} does not exist.", directory);
    }
    return fs::is_empty(directory);
}

bool MakeDirectory(const std::string &directory) {
//#ifdef WINDOWS
#ifdef _WIN32
    return (_mkdir(directory.c_str()) == 0);
#else
    return (mkdir(directory.c_str(), S_IRWXU) == 0);
#endif
}

bool MakeDirectoryHierarchy(const std::string &directory) {
    std::string full_path = GetRegularizedDirectoryName(directory);
    size_t curr_pos = full_path.find_first_of("/\\", 1);
    while (curr_pos != std::string::npos) {
        std::string subdir = full_path.substr(0, curr_pos + 1);
        if (!DirectoryExists(subdir)) {
            if (!MakeDirectory(subdir)) {
                return false;
            }
        }
        curr_pos = full_path.find_first_of("/\\", curr_pos + 1);
    }
    return true;
}

bool DeleteDirectory(const std::string &directory) {
    std::error_code error;
    if (fs::remove_all(directory, error) == static_cast<std::uintmax_t>(-1)) {
        LOG_WARN("Failed to remove directory {}", directory);
        return false;
    }
    return true;
}

bool FileExists(const std::string &filename) {
    return fs::exists(filename) && fs::is_regular_file(filename);
}

bool RemoveFile(const std::string &filename) {
    return (std::remove(filename.c_str()) == 0);
}

bool ListDirectory(const std::string &directory,
                   std::vector<std::string> &subdirs,
                   std::vector<std::string> &filenames) {
    if (directory.empty()) {
        return false;
    }
    DIR *dir;
    struct dirent *ent;
    struct stat st;
    dir = opendir(directory.c_str());
    if (!dir) {
        return false;
    }
    filenames.clear();
    while ((ent = readdir(dir)) != NULL) {
        const std::string file_name = ent->d_name;
        if (file_name[0] == '.') continue;
        std::string full_file_name =
                GetRegularizedDirectoryName(directory) + file_name;
        if (stat(full_file_name.c_str(), &st) == -1) continue;
        if (S_ISDIR(st.st_mode))
            subdirs.push_back(full_file_name);
        else if (S_ISREG(st.st_mode))
            filenames.push_back(full_file_name);
    }
    closedir(dir);
    return true;
}

bool ListFilesInDirectory(const std::string &directory,
                          std::vector<std::string> &filenames) {
    std::vector<std::string> subdirs;
    return ListDirectory(directory, subdirs, filenames);
}

bool ListFilesInDirectoryWithExtension(const std::string &directory,
                                       const std::string &extname,
                                       std::vector<std::string> &filenames) {
    std::vector<std::string> all_files;
    if (!ListFilesInDirectory(directory, all_files)) {
        return false;
    }
    std::string ext_in_lower = extname;
    std::transform(ext_in_lower.begin(), ext_in_lower.end(),
                   ext_in_lower.begin(), ::tolower);
    filenames.clear();
    for (const auto &fn : all_files) {
        if (GetFileExtensionInLowerCase(fn) == ext_in_lower) {
            filenames.push_back(fn);
        }
    }
    return true;
}

std::vector<std::string> FindFilesRecursively(
        const std::string &directory,
        std::function<bool(const std::string &)> is_match) {
    std::vector<std::string> matches;

    std::vector<std::string> subdirs;
    std::vector<std::string> files;
    ListDirectory(directory, subdirs, files);  // results are paths
    for (auto &f : files) {
        if (is_match(f)) {
            matches.push_back(f);
        }
    }
    for (auto &d : subdirs) {
        auto submatches = FindFilesRecursively(d, is_match);
        if (!submatches.empty()) {
            matches.insert(matches.end(), submatches.begin(), submatches.end());
        }
    }

    return matches;
}

FILE *FOpen(const std::string &filename, const std::string &mode) {
    FILE *fp;
#ifndef _WIN32
    fp = fopen(filename.c_str(), mode.c_str());
#else
    std::wstring filename_w;
    filename_w.resize(filename.size());
    int newSize = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(),
                                      static_cast<int>(filename.length()),
                                      const_cast<wchar_t *>(filename_w.c_str()),
                                      static_cast<int>(filename.length()));
    filename_w.resize(newSize);
    std::wstring mode_w(mode.begin(), mode.end());
    fp = _wfopen(filename_w.c_str(), mode_w.c_str());
#endif
    return fp;
}

std::string JoinPath(const std::string &path_component1,
                     const std::string &path_component2) {
    fs::path path(path_component1);
    return (path / path_component2).string();
}

std::string JoinPath(const std::vector<std::string> &path_components) {
    fs::path path;
    for (const auto &pc : path_components) {
        path /= pc;
    }
    return path.string();
}

}  // namespace filesystem
}  // namespace utility
}  // namespace hymson3d