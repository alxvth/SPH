#pragma once

#include "Logger.hpp"

#include <chrono>       // high_resolution_clock, time_point, milliseconds, duration_cast
#include <functional>   // function
#include <string>       // string, to_string

namespace sph::utils {

    /// ////// ///
    /// TIMING ///
    /// ////// ///

    using clock = std::chrono::high_resolution_clock;

    inline auto now()
    {
        return clock::now();
    }

    template<typename duration = std::chrono::milliseconds>
    inline auto timeSince(const clock::time_point& timePoint)
    {
        return std::chrono::duration_cast<duration>(clock::now() - timePoint).count();
    }

    /* Logs the time of a lambda function in microseconds, call like:
    *
        utils::timer([&]() {
             <CODE YOU WANT TO TIME>
            },
            "<DESCRIPTION>");
    */
    template <typename F>
    void timer(F myFunc, const std::string& name) {
        const auto time_start = now();
        myFunc();
        Log::info("Timing {0}: {1} microseconds", name, std::to_string(timeSince<std::chrono::microseconds>(time_start)));
    }

    template <typename T>
    concept IsStdDuration = requires {
        [] <class Rep, class Period>(std::type_identity<std::chrono::duration<Rep, Period>>) {}(
            std::type_identity<T>());
    };

    template<typename Resolution>
        requires IsStdDuration<Resolution>
    class ScopedTimerBase {
        using clock = utils::clock;
    public:
        //! start the timer
        ScopedTimerBase(const std::string& title, std::function<void(const std::string&)> logFunc, const std::string& secString) :
            _title(title), 
            _secString(secString), 
            _logFunc(logFunc), 
            _start(clock::now())
        {
        }

        //! stop the timer and save the elapsedTime
        ~ScopedTimerBase() {
            auto duration = std::chrono::duration_cast<Resolution>(clock::now() - _start).count();
            _logFunc(fmt::format("Duration of {0}: {1} {2}", _title, duration, _secString));
        }
         
    private:
        std::string _title;
        std::string _secString;
        std::function<void(std::string)> _logFunc;
        std::chrono::time_point<clock> _start;
    };

    /* Logs the time of a scope, call like:
    *
        {
            utils::ScopedTimer myTimer("My Scope");
            <CODE YOU WANT TO TIME>
        }
        
        {
            utils::ScopedTimer<>std::chrono::seconds myTimer("My Scope 2", "sec");
            <CODE YOU WANT TO TIME>
        }

        // prints:
        // Duration of My Scope: 99 ms"
        // Duration of My Scope 2: 4 sec"
    */
    template<typename Resolution = std::chrono::milliseconds>
        requires IsStdDuration<Resolution>
    class ScopedTimer : public ScopedTimerBase<Resolution> {
    public:
        ScopedTimer(const std::string& title, const std::string& secString = "ms") :
            ScopedTimerBase<Resolution>(title, static_cast<void(*)(const std::string&)>(Log::info), secString)
        {
        }

        ~ScopedTimer() = default;
    };

    template<typename Resolution = std::chrono::milliseconds>
        requires IsStdDuration<Resolution>
    class ScopedTimerDebug : public ScopedTimerBase<Resolution> {
    public:
        ScopedTimerDebug(const std::string& title, const std::string& secString = "ms") :
            ScopedTimerBase<Resolution>(title, static_cast<void(*)(const std::string&)>(Log::debug), secString)
        {
        }

        ~ScopedTimerDebug() = default;
    };


} // namespace sph::utils
