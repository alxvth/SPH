#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

#include <spdlog/common.h>
#include <spdlog/logger.h>

// Use like:
//      #include "Logger.h"
//      Log::info("Important message");
//      Log::debug("Very {0} {1} messages", "helpful", 2);
namespace sph::utils
{
    namespace fs = std::filesystem;

// Singleton logger class
// Logs to both console and file
// 
// spdlog source: https://github.com/gabime/spdlog
// spdlog docs:   https://spdlog.docsforge.com/
class Logger {
public:
    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;

    static Logger& getInstance()
    {
        static Logger logger;
        return logger;
    }

    static void setLogPath(const fs::path& logPath) {
        _log_file_path = logPath;
    }

    // lowest level log (0)
    static void trace(const std::string& message)
    {
        getInstance()._logger->trace(message);
    }

    // debug level log (1)
    static void debug(const std::string& message)
    {
        getInstance()._logger->debug(message);
    }

    // standard level log (2)
    static void info(const std::string& message)
    {
        getInstance()._logger->info(message);
    }

    // warning level log (3)
    static void warn(const std::string& message)
    {
        getInstance()._logger->warn(message);
    }

    // error level log (4)
    static void error(const std::string& message)
    {
        getInstance()._logger->error(message);
    }

    // critical level log (5)
    static void critical(const std::string& message)
    {
        getInstance()._logger->critical(message);
    }

    // set to "off" (6) to not log
    static void set_level(spdlog::level::level_enum log_level) {
        getInstance()._logger->set_level(log_level);

        for(auto& sink: getInstance()._logger->sinks())
            sink->set_level(log_level);
    }

    // redirect std cout, clog and cerr to this logger
    static void redirect_std_io_to_logger();

    // reset cout, clog and cerr to previous output (by default, console)
    static void reset_std_io(bool verbose = true);

    // Return path of log file
    std::string getLogFilePath() const
    {
        return getInstance()._log_file_path.string();
    }

    void static flush()
    {
        getInstance()._logger->flush();
    }

private:

    // Helper class for redirecting std io to this Logger, see https://en.cppreference.com/w/cpp/io/basic_streambuf
    class RedirectLog: public std::streambuf
    {
    public:
        RedirectLog(std::ostream& o, const spdlog::level::level_enum& sdplevel);
        ~RedirectLog();

    public:
        int overflow(int c) override;
        int sync(void) override;

    private:
        std::ostream& _std_out;
        std::streambuf* const _std_out_buf;
        std::string _sbuffer;
        std::function<void(const std::string& msg)> _log_fn;
    };

    // Private constructor
    Logger();

    // Logger singleton instance
    std::unique_ptr<spdlog::logger> _logger;

    // Log file name, dir and path
    static fs::path _log_file_name;
    static fs::path _log_file_dir;
    static fs::path _log_file_path;

    // Pointer to std io redirection handler
    std::array<std::unique_ptr<RedirectLog>, 3> _redirectLogs;
};

} // namespace sph::utils

namespace sph::Log {

    // lowest level log (0)
    template <typename... T>
    inline void trace(const std::string& message, T&&... args) { sph::utils::Logger::trace(fmt::vformat(message, fmt::make_format_args(args...))); }

    // lowest level log (0)
    inline void trace(const std::string& message){ sph::utils::Logger::trace(message); }

    // debug level log (1)
    template <typename... T>
    inline void debug(const std::string& message, T&&... args) { sph::utils::Logger::debug(fmt::vformat(message, fmt::make_format_args(args...))); }

    // debug level log (1)
    inline void debug(const std::string& message){ sph::utils::Logger::debug(message); }

    // standard level log (2)
    template <typename... T>
    inline void info(const std::string& message, T&&... args){ sph::utils::Logger::info(fmt::vformat(message, fmt::make_format_args(args...))); }

    // standard level log (2)
    inline void info(const std::string& message){ sph::utils::Logger::info(message); }

    // warning level log (3)
    template <typename... T>
    inline void warn(const std::string& message, T&&... args) { sph::utils::Logger::warn(fmt::vformat(message, fmt::make_format_args(args...))); }

    // warning level log (3)
    inline void warn(const std::string& message){ sph::utils::Logger::warn(message); }

    // error level log (4)
    template <typename... T>
    inline void error(const std::string& message, T&&... args) { sph::utils::Logger::error(fmt::vformat(message, fmt::make_format_args(args...))); }

    // error level log (4)
    inline void error(const std::string& message){ sph::utils::Logger::error(message); }

    // critical level log (5)
    template <typename... T>
    inline void critical(const std::string& message, T&&... args) { sph::utils::Logger::critical(fmt::vformat(message, fmt::make_format_args(args...))); }

    // critical level log (5)
    inline void critical(const std::string& message){ sph::utils::Logger::critical(message); }

    // set to "off" (6) to not log
    inline void set_level(spdlog::level::level_enum log_level) { sph::utils::Logger::set_level(log_level); }

    // redirect std cout, clog and cerr to this logger
    inline void redirect_std_io_to_logger() { sph::utils::Logger::redirect_std_io_to_logger();  }

    // reset cout, clog and cerr to previous output (by default, console)
    inline void reset_std_io(bool verbose = true) { sph::utils::Logger::reset_std_io(verbose); }

} // namespace sph::Log
