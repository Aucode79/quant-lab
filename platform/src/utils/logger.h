#pragma once
/**
 * 轻量级日志工具
 *
 * 基于 {fmt} 库，提供带时间戳和级别的格式化输出。
 */

#include <string>
#include <chrono>
#include <iostream>
#include <mutex>

#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/format.h>

namespace quant::util {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) { level_ = level; }

    template<typename... Args>
    void log(LogLevel level, fmt::format_string<Args...> fmt_str, Args&&... args) {
        if (level < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::system_clock::now();
        auto time_str = fmt::format("{:%H:%M:%S}", fmt::localtime(
            std::chrono::system_clock::to_time_t(now)));

        fmt::text_style style;
        const char* level_str;

        switch (level) {
            case LogLevel::TRACE:
                style = fmt::fg(fmt::color::gray);
                level_str = "TRACE"; break;
            case LogLevel::DEBUG:
                style = fmt::fg(fmt::color::cyan);
                level_str = "DEBUG"; break;
            case LogLevel::INFO:
                style = fmt::fg(fmt::color::green);
                level_str = " INFO"; break;
            case LogLevel::WARN:
                style = fmt::fg(fmt::color::yellow);
                level_str = " WARN"; break;
            case LogLevel::ERROR:
                style = fmt::fg(fmt::color::red);
                level_str = "ERROR"; break;
        }

        fmt::print(style, "[{} {}] ", time_str, level_str);
        fmt::print(fmt_str, std::forward<Args>(args)...);
        fmt::print("\n");
    }

private:
    Logger() = default;
    LogLevel level_ = LogLevel::INFO;
    std::mutex mutex_;
};

// 便捷宏
#define LOG_TRACE(...) Logger::instance().log(LogLevel::TRACE, __VA_ARGS__)
#define LOG_DEBUG(...) Logger::instance().log(LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...)  Logger::instance().log(LogLevel::INFO,  __VA_ARGS__)
#define LOG_WARN(...)  Logger::instance().log(LogLevel::WARN,  __VA_ARGS__)
#define LOG_ERROR(...) Logger::instance().log(LogLevel::ERROR, __VA_ARGS__)

}  // namespace quant::util
