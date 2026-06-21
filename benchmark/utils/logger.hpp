// Logger class.

#ifndef _LOGGER_HPP_
#define _LOGGER_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>
#include <vector>

class Logger {
private:
    std::ofstream logFile;

    // Get the current working directory.
    std::string getCurrentDirectory() {
        char buffer[1024];
        if (!getcwd(buffer, sizeof(buffer))) {
            throw std::runtime_error("Failed to get current working directory");
        }
        return std::string(buffer);
    }

    // Get the executable path.
    std::string getExecutablePath() {
        char buffer[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
        if (count == -1) {
            throw std::runtime_error("Failed to get executable path");
        }
        buffer[count] = '\0';  // Null-terminate the string
        return std::string(buffer);
    }

    // Extract the executable file name.
    std::string getExecutableName(const std::string& execPath) {
        size_t lastSlash = execPath.find_last_of('/');
        if (lastSlash == std::string::npos) {
            return execPath;  // If the path does not contain `/`, return the file name directly.
        }
        return execPath.substr(lastSlash + 1);  // Extract the final file name.
    }

    // Check whether the directory exists.
    bool directoryExists(const std::string& path) {
        struct stat info;
        return (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR));
    }

    // Create a directory.
    void createDirectory(const std::string& path) {
        if (!directoryExists(path)) {
            if (mkdir(path.c_str(), 0755) != 0) {
                throw std::runtime_error("Failed to create directory: " + path);
            }
        }
    }

    // Get the log file path.
    std::string getLogFilePath(const std::string& currentDir, const std::string& execName) {
        // Assume the current directory is base_dir/benchmark/XX.
        size_t lastSlash = currentDir.find_last_of('/');
        size_t secondLastSlash = currentDir.find_last_of('/', lastSlash - 1);

        if (lastSlash == std::string::npos || secondLastSlash == std::string::npos) {
            throw std::runtime_error("Invalid working directory structure: " + currentDir);
        }

        // Extract base_dir and XX.
        std::string baseDir = currentDir.substr(0, secondLastSlash);
        std::string xx = currentDir.substr(currentDir.find_last_of('/') + 1);

        std::string resultsDir = baseDir + "/results/" + xx;

        // Ensure the results/XX directory exists.
        createDirectory(baseDir + "/results");
        createDirectory(resultsDir);

        // Return the log file path.
        return resultsDir + "/" + execName + ".txt";
    }

public:
    Logger(std::string file_name="", bool append=false) {
        // Automatically get the current directory and executable name.
        std::string currentDir = getCurrentDirectory();
        std::string execPath = getExecutablePath();
        std::string execName;
        if (file_name=="")
            execName = getExecutableName(execPath);
        else
            execName = file_name;

        std::string logFilePath = getLogFilePath(currentDir, execName);

        if(append)
            logFile.open(logFilePath, std::ios::out | std::ios::app);
        else
            logFile.open(logFilePath, std::ios::out | std::ios::trunc);
        if (!logFile.is_open()) {
            throw std::runtime_error("Failed to open log file: " + logFilePath);
        }
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    void logging(const std::string& message, bool change_line=true, bool verbose = false) {
        // Write to the log file.
        if (logFile.is_open()) {
            logFile << message;
            if(change_line)
                logFile << "\n\n";
            logFile.flush();
        }

        // If verbose is true, also print to the command line.
        if (verbose) {
            std::cout << message << std::endl;
            std::cout.flush();
        }
    }

    // Output logs with printf-style formatting.
    template<typename... Args>
    void loggingf(const char* format, bool change_line, bool verbose, Args... args) {
        // Estimate the buffer size first.
        int size = std::snprintf(nullptr, 0, format, args...) + 1; // +1 for '\0'
        if (size <= 0) {
            // If formatting fails, output the format string directly.
            logging(format, change_line, verbose);
            return;
        }
        std::vector<char> buf(size);
        std::snprintf(buf.data(), size, format, args...);
        logging(std::string(buf.data()), change_line, verbose);
    }

    // Overload for convenience; defaults are change_line=true and verbose=false.
    template<typename... Args>
    void loggingf(const char* format, Args... args) {
        loggingf(format, false, false, args...);
    }

    void flush_log(){
        logFile.flush();
    }

};


#endif