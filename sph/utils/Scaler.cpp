#include "Scaler.hpp"

#include "Algorithms.hpp"
#include "CommonDefinitions.hpp"
#include "Data.hpp"
#include "Logger.hpp"
#include "Math.hpp"
#include "Settings.hpp"

#include <algorithm>
#include <execution>

namespace sph::utils {

    void scale(Data& data, const Scaler scaler)
    {
        vf32& dataRef = data.getData();

        switch (scaler)
        {
        case sph::utils::Scaler::NONE:
            Log::info("scale data with NONE: doing nothing");
            break;
        case sph::utils::Scaler::STANDARD:
            Log::info("scale data with STANDARD: z = (x - u) / s [channel-wise]");
            normalizeStandard(dataRef, data.getNumDimensions());
            break;
        case sph::utils::Scaler::UNIFORM:
            Log::info("scale data with UNIFORM: z = x / max [channel-wise]");
            normalizeChannelsUniform(dataRef, data.getNumDimensions());
            break;
        case sph::utils::Scaler::ROBUST:
        {
            Log::info("scale data with ROBUST: clamps data to 95% and normalizes values to [0, 1] [globally]");
            // clip to [0, 95-percentile]
            const float quantile95 = computeQuantile(dataRef, 0.95f, {}, 1);
            clampValues(dataRef, 0.f, quantile95);

            // Globally normalize to [0, 1]
            std::transform(SPH_PARALLEL_EXECUTION
                dataRef.begin(), dataRef.end(), // from, to
                dataRef.begin(),                // write to the same location
                [quantile95](float val) { return val / quantile95; });
        }
            break;
        }
    }

} // sph::utils