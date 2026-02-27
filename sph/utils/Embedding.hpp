#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace sph::utils {

    struct Hierarchy;

    /// ////////// ///
    /// EMBEDDINGS ///
    /// ////////// ///
    class EmbeddingExtends
    {
    public:
        EmbeddingExtends() = default;
        EmbeddingExtends(float x_min, float x_max, float y_min, float y_max);   // be sure that x_max >= x_min and y_max >= y_min

        void setExtends(const EmbeddingExtends& extends);                       // be sure that x_max >= x_min and y_max >= y_min
        void setExtends(float x_min, float x_max, float y_min, float y_max);    // be sure that x_max >= x_min and y_max >= y_min

        float x_min() const { return _x_min; }
        float x_max() const { return _x_max; }
        float y_min() const { return _y_min; }
        float y_max() const { return _y_max; }
        float extend_x() const { return _extend_x; }
        float extend_y() const { return _extend_y; }

        std::string getMinMaxString() const;

    private:
        float _x_min = 0.f;
        float _x_max = 0.f;
        float _y_min = 0.f;
        float _y_max = 0.f;
        float _extend_x = 0.f;
        float _extend_y = 0.f;
    };

    EmbeddingExtends computeExtends(const std::vector<float>& emb);

    void scaleEmbeddingToStd(std::vector<float>& emb, const float stdDevDesired = 0.0001f);

    void scaleEmbeddingToOne(std::vector<float>& emb);

    std::vector<float> averageEmbeddingPositionOfChildren(const Hierarchy& hierarchy, const std::vector<float>& embedding, const size_t currentLevel);

    // emb has to be of size numEmbPoints * 2
    void randomEmbeddingInit(std::vector<float>& emb, float x, float y);

} // namespace sph::utils

std::ostream& operator<<(std::ostream& os, const sph::utils::EmbeddingExtends& ext);
std::string operator<<(const std::string& os, const sph::utils::EmbeddingExtends& ext);

