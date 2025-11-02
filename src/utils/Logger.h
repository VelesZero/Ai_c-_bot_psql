#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <memory>
#include <sstream>
#include <chrono>
#include <iomanip>

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

class Logger {
public:
    static Logger& getInstance();
    
    void setLogFile(const std::string& filename);
    void setLogLevel(LogLevel level);
    
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    void critical(const std::string& message);
    
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger();
    ~Logger();
    
    void log(LogLevel level, const std::string& message);
    std::string getCurrentTime();
    std::string levelToString(LogLevel level);
    
    std::ofstream logFile_;
    LogLevel currentLevel_;
    std::mutex mutex_;
};

#endif // LOGGER_H