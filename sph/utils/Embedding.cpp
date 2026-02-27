#include "Embedding.hpp"

#include "Hierarchy.hpp"
#include "Logger.hpp"
#include "Math.hpp"

#include <cassert>
#include <cmath>

#include <fmt/format.h>

namespace sph::utils {

    /// ////////// ///
    /// EMBEDDINGS ///
    /// ////////// ///

    EmbeddingExtends::EmbeddingExtends(float x_min, float x_max, float y_min, float y_max) : _x_min(x_min), _x_max(x_max), _y_min(y_min), _y_max(y_max)
    {
        _extend_x = _x_max - _x_min;
        _extend_y = _y_max - _y_min;
    }

    void EmbeddingExtends::setExtends(float x_min, float x_max, float y_min, float y_max) {
        if (!(x_min < x_max))
            Log::warn("EmbeddingExtends::setExtends: x_min < x_max");
        if (!(y_min < y_max))
            Log::warn("EmbeddingExtends::setExtends: y_min < y_max");

        _x_min = x_min; _x_max = x_max;
        _y_min = y_min; _y_max = y_max;

        _extend_x = _x_max - _x_min;
        _extend_y = _y_max - _y_min;
    }

    void EmbeddingExtends::setExtends(const EmbeddingExtends& extends) {
        setExtends(extends.x_min(), extends.x_max(), extends.y_min(), extends.y_max());
    }

    std::string EmbeddingExtends::getMinMaxString() const {
        return fmt::format("x in [{0}, {1}], y in [{2}, {3}]", _x_min, _x_max, _y_min, _y_max);
    }

    EmbeddingExtends computeExtends(const std::vector<float>& emb)
    {
        float x_min(0), x_max(0), y_min(0), y_max(0);

        auto data = emb.data();
        for (size_t i = 0; i < emb.size() / 2; i++)
        {
            if (x_min > data[i * 2]) x_min = data[i * 2];
            if (x_max < data[i * 2]) x_max = data[i * 2];
            if (y_min > data[i * 2 + 1]) y_min = data[i * 2 + 1];
            if (y_max < data[i * 2 + 1]) y_max = data[i * 2 + 1];
        }

        return { x_min, x_max, y_min, y_max };
    }

    void scaleEmbeddingToStd(std::vector<float>& emb, const float stdDevDesired /* = 0.0001f */)
    {
        const int64_t numPoints = emb.size() / 2;

        // Calculate the mean and standard deviation of the first embedding dimension
        float sum = 0.f;
        for (int64_t i = 0; i < numPoints; ++i)
            sum += emb[i * 2];

        const float mean = sum / numPoints;

        float stdDevCurrent = 0.f;
        for (int64_t i = 0; i < numPoints; ++i)
            stdDevCurrent += std::pow(emb[i * 2] - mean, 2.f);

        stdDevCurrent = std::sqrt(stdDevCurrent / numPoints);

        // Re-scale the data to match the desired standard deviation
        const float scaleFactor = stdDevDesired / stdDevCurrent;

        SPH_PARALLEL
        for (int64_t i = 0; i < numPoints; ++i) {
            emb[i * 2] = (emb[i * 2] - mean) * scaleFactor + mean;
            emb[i * 2 + 1] = (emb[i * 2 + 1] - mean) * scaleFactor + mean;
        }
    }

    void scaleEmbeddingToOne(std::vector<float>& emb)
    {
        const int64_t numPoints = emb.size() / 2;

        float sum = 0.f;
        for (int64_t i = 0; i < numPoints; ++i)
            sum += emb[i * 2];

        const float mean = sum / numPoints;

        SPH_PARALLEL
        for (int64_t i = 0; i < numPoints; ++i) {
            emb[i * 2] = (emb[i * 2] - mean);
            emb[i * 2 + 1] = (emb[i * 2 + 1] - mean);
        }

        const EmbeddingExtends currentExtends = computeExtends(emb);

        // Re-scale the data to match the desired standard deviation
        float scaleX = std::max(std::abs(currentExtends.x_min()), std::abs(currentExtends.x_max()));
        float scaleY = std::max(std::abs(currentExtends.y_min()), std::abs(currentExtends.y_max()));

        if (isBasicallyZero(scaleX))
            scaleX = 1.f;

        if (isBasicallyZero(scaleY))
            scaleY = 1.f;

        SPH_PARALLEL
        for (int64_t i = 0; i < numPoints; ++i) {
            emb[i * 2] = emb[i * 2] / scaleX;
            emb[i * 2 + 1] = emb[i * 2 + 1] / scaleY;
        }

#if SPH_DEBUG
        {
            const EmbeddingExtends newExtends = computeExtends(emb);
            assert(numPoints <= 1 || std::abs(std::max(std::abs(newExtends.x_min()), std::abs(newExtends.x_max())) - 1.f) <= 0.01f);
            assert(numPoints <= 1 || std::abs(std::max(std::abs(newExtends.y_min()), std::abs(newExtends.y_max())) - 1.f) <= 0.01f);
        }
#endif
    }

    std::vector<float> averageEmbeddingPositionOfChildren(const Hierarchy& hierarchy, const std::vector<float>& embedding, const size_t currentLevel)
    {
        assert(currentLevel > 0);

        const auto numPoints = hierarchy.numComponentsOn(currentLevel);
        const auto& children = hierarchy.childrenOn(currentLevel);

        std::vector<float> avgPos(numPoints * 2);

        assert(children.size() == numPoints);

        if (numPoints == 1)
            return avgPos;

        for (size_t emdId = 0; emdId < numPoints; emdId++)
        {
            const auto& childIDs = children[emdId];

            for (const auto& childID : childIDs) {
                avgPos[emdId * 2]        += embedding[childID * 2];
                avgPos[emdId * 2 + 1]    += embedding[childID * 2 + 1];
            }
        }

        for (size_t emdId = 0; emdId < numPoints; emdId++)
        {
            const auto numChildren = children[emdId].size();
            avgPos[emdId * 2] /= numChildren;
            avgPos[emdId * 2 + 1] /= numChildren;
        }

        return avgPos;
    }

    void randomEmbeddingInit(std::vector<float>& emb, float x, float y) {
        const int64_t numEmbPoints = emb.size() / 2;

        SPH_PARALLEL
        for (int64_t i = 0; i < numEmbPoints; i++)
        {
            const auto randomPoint = utils::randomVec(x, y);

            emb[2ll * i] = randomPoint.first;
            emb[2ll * i + 1] = randomPoint.second;
        }
    }

} // namespace sph::utils

std::ostream& operator<<(std::ostream& os, const sph::utils::EmbeddingExtends& ext) {
    return os << ext.getMinMaxString() << std::endl;
}

std::string operator<<(const std::string& os, const sph::utils::EmbeddingExtends& ext) {
    return os + ext.getMinMaxString();
}
