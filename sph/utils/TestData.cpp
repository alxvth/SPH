#include "TestData.hpp"

#include "CommonDefinitions.hpp"
#include "Timer.hpp"

#include <tinycolormap.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <numeric>
#include <random>

namespace sph::utils {

    constexpr int64_t DIMS = 3;
    constexpr float PI = std::numbers::pi_v<float>;

    constexpr int64_t X = 0;
    constexpr int64_t Y = 1;
    constexpr int64_t Z = 2;

    void createRandomData(std::vector<float>& data, const int64_t d, const int64_t nb)
    {
        utils::ScopedTimerDebug<> myTimer("createRandomData");

        data.clear();
        data.resize(static_cast<size_t>(d) * nb);

        std::mt19937_64 rng;
        std::uniform_real_distribution<float> distrib;

        SPH_PARALLEL
        for (int64_t i = 0; i < nb; i++) {
            for (int64_t j = 0; j < d; j++)
                data[static_cast<size_t>(d) * i + j] = distrib(rng);
        }

    }

    void createSwissRole(std::vector<float>& positions, std::vector<float>& colors, const int64_t n_samples, const float noise, const uint64_t random_state)
    {
        utils::ScopedTimerDebug<> myTimer("createSwissRole");

        auto norm = [](float val) -> double {
            return (val - (1.5f * PI)) / (3.0 * PI);
        };

        positions.clear();
        positions.resize(3ll * n_samples);

        colors.clear();
        colors.resize(3ll * n_samples);

        std::random_device rd;
        std::mt19937_64 gen(rd());
        gen.seed(random_state);

        std::uniform_real_distribution<float> uniformDist (0.f, 1.f);
        std::normal_distribution<float> normNoise (0.f, 1.f);

        SPH_PARALLEL
        for (int64_t sample = 0; sample < n_samples; sample++) {
            const int64_t pos = sample * DIMS;
            const float rand = 1.5f * PI * (1.f + 2.f * uniformDist(gen));

            // color according to distance from center (z-axis)
            tinycolormap::Color color = tinycolormap::GetColor(norm(rand), tinycolormap::ColormapType::Viridis);
            std::transform(color.data, color.data + 3, colors.begin() + pos, [](const double value) {
                return static_cast<float>(value);
                });

            positions[pos + Z] = 21 * uniformDist(gen);    // uniformly distributed along z-axis
            positions[pos + Y] = rand * std::sin(rand);     // outward spiral
            positions[pos + X] = rand * std::cos(rand);     // outward spiral

            if (noise != 0)
            {
                positions[pos + X] += noise * normNoise(gen);
                positions[pos + Y] += noise * normNoise(gen);
                positions[pos + Z] += noise * normNoise(gen);
            }
        }

    }

    void createSCurve(std::vector<float>& positions, std::vector<float>& colors, const int64_t n_samples, const float noise, const uint64_t random_state)
    {
        utils::ScopedTimerDebug<> myTimer("createSCurve");

        auto norm = [](float val) -> double {
            return (val + 1.5 * PI) / (3.0 * PI);
        };

        auto sign = [](float val) -> int {
            return (0.f < val) - (val < 0.f);
        };

        positions.clear();
        positions.resize(3ll * n_samples);

        colors.clear();
        colors.resize(3ll * n_samples);

        std::random_device rd;
        std::mt19937_64 gen = std::mt19937_64(rd());
        gen.seed(random_state);

        std::uniform_real_distribution<float> uniformDist(0.f, 1.f);
        std::normal_distribution<float> normNoise(0.f, 1.f);

        SPH_PARALLEL
        for (int64_t sample = 0; sample < n_samples; sample++) {
            const int64_t pos = sample * DIMS;
            const float rand = 3.f * PI * (uniformDist(gen) - 0.5f);

            // color according to distance from center (z-axis)
            tinycolormap::Color color = tinycolormap::GetColor(norm(rand), tinycolormap::ColormapType::Viridis);
            std::transform(color.data, color.data + 3, colors.begin() + pos, [](const double value) {
                return static_cast<float>(value);
                });

            positions[pos + X] = std::sin(rand); 
            positions[pos + Y] = sign(rand) * (std::cos(rand) - 1.f);
            positions[pos + Z] = 2.f * uniformDist(gen);

            if (noise != 0)
            {
                positions[pos + X] += noise * normNoise(gen);
                positions[pos + Y] += noise * normNoise(gen);
                positions[pos + Z] += noise * normNoise(gen);
            }
        }

    }
    
    void create3dGaussians(std::vector<float>& positions, std::vector<float>& colors, const int64_t n_samples, const float noise, const uint64_t random_state, const std::vector<float> centers)
    {
        assert(centers.size() % 3 == 0);    // we want 3d coordinates

        utils::ScopedTimerDebug<> myTimer("create3dGaussians");

        auto norm = [](const float x, const float y, const float z, const float center_x, const float center_y, const float center_z) -> float {
            return std::sqrt(std::pow(x - center_x, 2.f) + std::pow(y - center_y, 2.f) + std::pow(z - center_z, 2.f));
            };

        const int64_t numGaussians = centers.size() / 3;

        const int pointsPerGaussian   = static_cast<int>(n_samples / numGaussians);
        const int remainder           = static_cast<int>(n_samples % numGaussians);

        positions.clear();
        positions.resize(3ll * n_samples);

        colors.clear();
        colors.resize(3ll * n_samples);

        std::random_device rd;
        std::mt19937_64 gen = std::mt19937_64(rd());
        gen.seed(random_state);

        std::normal_distribution<float> normNoise(0.f, 1.f);

        auto populateSphere = [&positions, &colors, &centers, &noise, &pointsPerGaussian, &normNoise, &gen, &norm](const int numSamples, const int64_t sphereID) {

            std::normal_distribution<float> distX(centers[sphereID * DIMS], 1.f);
            std::normal_distribution<float> distY(centers[sphereID * DIMS + 1], 1.f);
            std::normal_distribution<float> distZ(centers[sphereID * DIMS + 2], 1.f);

            SPH_PARALLEL
            for (int sample = 0; sample < numSamples; sample++) {
                const int64_t pos = pointsPerGaussian * sphereID * 3 + sample * DIMS;

                positions[pos + X] = distX(gen);
                positions[pos + Y] = distY(gen);
                positions[pos + Z] = distZ(gen);

                // color according to distance from center
                const auto pointDistToCenter = norm(positions[pos + X], positions[pos + Y], positions[pos + Z], centers[sphereID * DIMS], centers[sphereID * DIMS + 1], centers[sphereID * DIMS + 2]);
                tinycolormap::Color color = tinycolormap::GetColor(pointDistToCenter / 3, tinycolormap::ColormapType::Viridis);
                std::transform(color.data, color.data + 3, colors.begin() + pos, [](const double value) {
                    return static_cast<float>(value);
                    });

                if (noise != 0)
                {
                    positions[pos + X] += noise * normNoise(gen);
                    positions[pos + Y] += noise * normNoise(gen);
                    positions[pos + Z] += noise * normNoise(gen);
                }
            }

            };

        for (int64_t sphereID = 0; sphereID < numGaussians; sphereID++)
            populateSphere(pointsPerGaussian, sphereID);


        if(remainder > 0)
            populateSphere(remainder, numGaussians - 1);

    }

} // namespace sph::utils
