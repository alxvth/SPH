#include "Logger.hpp"

#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>

#include <spdlog/sinks/basic_file_sink.h>       // spdlog, sink type
#include <spdlog/sinks/stdout_color_sinks.h>    // spdlog, sink type

namespace sph::utils
{
    fs::path Logger::_log_file_name     = "SPH_log.txt";
    fs::path Logger::_log_file_dir      = fs::current_path();
    fs::path Logger::_log_file_path     = _log_file_dir / _log_file_name;

    Logger::Logger() {
        // set up console sink
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%H:%M:%S.%e] [%^%L%$] %v");

        // set up file sink
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(_log_file_path.string(), true);
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");

        // In debug mode, log more. 
#if SPH_RELEASE
        console_sink->set_level(spdlog::level::info);
        file_sink->set_level(spdlog::level::info);
#else
        console_sink->set_level(spdlog::level::debug);
        file_sink->set_level(spdlog::level::debug);
#endif

        // set sup logger
        _logger = std::make_unique<spdlog::logger>("Spatial Hierarchy (SPH) logger", spdlog::sinks_init_list{ console_sink, file_sink });
        _logger->set_level(spdlog::level::debug);

        // log yourself
        _logger->info("Logger set up. Log to console and file at " + _log_file_path.string());
    }

    void Logger::redirect_std_io_to_logger()
    {
        Logger::info("Redirect std io to this logger");

        Logger::reset_std_io(false);

        Logger& log = getInstance();
        log._redirectLogs[0] = std::make_unique<RedirectLog>(std::cout, spdlog::level::info);
        log._redirectLogs[1] = std::make_unique<RedirectLog>(std::clog, spdlog::level::debug);
        log._redirectLogs[2] = std::make_unique<RedirectLog>(std::cerr, spdlog::level::err);
    }

    void Logger::reset_std_io(bool verbose)
    {
        Logger& log = getInstance();
        log._redirectLogs[0].reset();
        log._redirectLogs[1].reset();
        log._redirectLogs[2].reset();

        if (verbose)
            Logger::info("Reset std io");
    }

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

    Logger::RedirectLog::RedirectLog(std::ostream& o, const spdlog::level::level_enum& sdplevel) :
        _std_out(o),
        _std_out_buf(o.rdbuf())
    {
        // Redirect std stream to streamlog
        _std_out.rdbuf(this);

        // Set output level
        switch (sdplevel)
        {
        case spdlog::level::err:
            _log_fn = std::bind(Logger::error, std::placeholders::_1);
            break;
        case spdlog::level::debug:
            _log_fn = std::bind(Logger::debug, std::placeholders::_1);
            break;
        case spdlog::level::info:
        default:
            _log_fn = std::bind(Logger::info, std::placeholders::_1);
            break;
        }
    }

    Logger::RedirectLog::~RedirectLog()
    {
        // Reset output
        _std_out.rdbuf(_std_out_buf);
    }

    int Logger::RedirectLog::overflow(int c)
    {
        // New incoming characters
        auto s = std::char_traits<char>::to_char_type(c);
        _sbuffer += s;
        return 0;
    }

    int Logger::RedirectLog::sync(void)
    {
        // Call actual logger function
        if (not _sbuffer.empty()) {
            _log_fn(_sbuffer);
            _sbuffer.clear();
        }

        return 0;
    }

} // namespace sph::utils
