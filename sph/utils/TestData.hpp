#pragma once

#include <cstdint>
#include <vector>

namespace sph::utils {
    
    enum class TestData : uint32_t
    {
        SwissRole,
        SCurve,
        Gaussians3D,
    };

    void createRandomData(std::vector<float>& data, int64_t d = 64, int64_t nb = 100'000);

    // noise is sampled from a normal distribution
    void createSwissRole(std::vector<float>& positions, std::vector<float>& colors, const int64_t n_samples = 1500, const float noise = 0.00, const uint64_t random_state = 1234);

    // noise is sampled from a normal distribution
    void createSCurve(std::vector<float>& positions, std::vector<float>& colors, const int64_t n_samples = 1500, const float noise = 0.00, const uint64_t random_state = 1234);

    // noise is sampled from a normal distribution
    void create3dGaussians(std::vector<float>& positions, std::vector<float>& colors, const int64_t n_samples = 1500, const float noise = 0.00, const uint64_t random_state = 1234, const std::vector<float> centers = {0.f, 0.f, 0.f});

} // namespace sph::utils
