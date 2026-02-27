#pragma once

#include <cstdint>
#include <mutex>

#include <fmt/base.h>

namespace sph::utils {

    /// ///////////////// ///
    /// For Loop Progress ///
    /// ///////////////// ///

    // Alternative could be https://github.com/gipert/progressbar (MIT)
    class ProgressBar {
    public:
        inline ProgressBar(size_t n, bool v = true) noexcept;

        /* Increments the progressCounter by 1
        When using with pragma omp parallel for:
            SPH_PARALLEL_CRITICAL
            progress.update();
        */
        inline void update() noexcept;

        /* Increments the progressCounter to newCounterVal */
        inline void update(uint64_t newCounterVal) noexcept;

        /* Increments the progressCounter by increaseVal */
        inline void updateBy(uint64_t increaseVal) noexcept;

        /* Increments the progressCounter by increaseVal and uses a std::mutex to be thread safe */
        inline void updateBySafe(uint64_t increaseVal) noexcept;

        /* Increments the progressCounter by 1 and uses a std::mutex to be thread safe */
        inline void updateSafe() noexcept;

        /* Increments the progressCounter to newCounterVal and uses a std::mutex to be thread safe */
        inline void updateSafe(uint64_t newCounterVal) noexcept;

        /* Sets progress to 100% */
        inline void finish() noexcept;

        /* Prints a line ending */
        inline void end() const noexcept;

        inline void reset(size_t n = 0) noexcept;

        inline void setVerbose(bool v) noexcept 
        { 
            verbose = v;
        };

        inline void print() noexcept;

    public:
        ProgressBar(ProgressBar const&)             = delete;
        ProgressBar& operator=(ProgressBar const&)  = delete;
        ProgressBar(ProgressBar&&)                  = delete;
        ProgressBar& operator=(ProgressBar&&)       = delete;

    private:
        std::mutex  m_progressCounter = {};
        uint64_t    progressCounter = 0;
        float       progress = 0.0f;
        float       step = 0.1f;
        size_t      numIter = 0;
        bool        verbose = true;
    };

    inline ProgressBar::ProgressBar(size_t n, bool v) noexcept :
        numIter(n),
        verbose(v)
    {
        print();                // print 0% progress
        std::fflush(stdout);    // flush first line
    }

    inline void ProgressBar::updateSafe() noexcept
    {
        if (!verbose)
            return;

        std::scoped_lock<std::mutex> lk(m_progressCounter);
        update();
    }

    inline void ProgressBar::updateSafe(uint64_t newCounterVal) noexcept
    {
        if (!verbose)
            return;

        std::scoped_lock<std::mutex> lk(m_progressCounter);
        update(newCounterVal);
    }

    inline void ProgressBar::updateBy(uint64_t increaseVal) noexcept
    {
        if (!verbose)
            return;

        update(progressCounter + increaseVal);
    }

    inline void ProgressBar::updateBySafe(uint64_t increaseVal) noexcept
    {
        if (!verbose)
            return;

        std::scoped_lock<std::mutex> lk(m_progressCounter);
        update(progressCounter + increaseVal);
    }

    inline void ProgressBar::update() noexcept
    {
        if (!verbose)
            return;

        update(progressCounter + 1);
    }

    inline void ProgressBar::update(uint64_t newCounterVal) noexcept
    {
        progressCounter = newCounterVal;

        if (!verbose)
            return;
        
        float newProgress = static_cast<float>(progressCounter) / numIter * 100.f;

        if(newProgress > progress + step )
        {
            progress = newProgress;
            print();
        }
    }

    // sets progress to 100% and prints a new line
    inline void ProgressBar::finish() noexcept
    {
        progressCounter = numIter;
        progress = 100.f;
        print();
        end();
    }

    // prints a new line
    inline void ProgressBar::end() const noexcept
    {
        if (!verbose)
            return;

        fmt::print("\n");
        std::fflush(stdout);
    }

    inline void ProgressBar::reset(size_t n /* = 0 */) noexcept
    {
        progressCounter = 0;
        progress = 0.f;

        if (n != 0)
            numIter = n;
    }

    inline void ProgressBar::print() noexcept
    {
        if (!verbose)
            return;

        fmt::print("\rProgress: {:.1f}% ({}/{})", progress, progressCounter, numIter);
    }

} // namespace sph::utils
