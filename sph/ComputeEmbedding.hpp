#pragma once

#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Graph.hpp>

#include "sph/EmbedTsne.hpp"
#include "sph/EmbedUmap.hpp"

#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

namespace sph {
    enum class EmbeddingType {
        TSNE,
        UMAP,
        COUNT // no embedding type, just used for numEmbeddingTypes
    };

    constexpr auto numEmbeddingTypes() {
        return static_cast<std::underlying_type_t<EmbeddingType>>(EmbeddingType::COUNT);
    }

    struct ComputeEmbeddingSettings {
        TsneEmbeddingParameters tsne = {};
        UmapEmbeddingParameters umap = {};
        float                   initRadius = 0.1f;
    };

    /*!
     *  computeEmbedding.initEmbedding(numCurrentEmbPoints);
     *  computeEmbedding.setSetting(embeddingSettings);
     *  computeEmbedding.computeTSNE(currentTransitionMatrix);
     *  auto emb = computeEmbedding.getEmbedding();
     */
    class ComputeEmbedding
    {
    public:
        ComputeEmbedding() = default;
        ~ComputeEmbedding() = default;

    public: // Initialization
        void initEmbedding([[maybe_unused]] uint64_t numEmbPoints, std::vector<float>&& embedding);
        void initEmbedding(uint64_t numEmbPoints);
        static void randomEmbeddingInit(std::vector<float>& emb, float x, float y);

    public: // Embedding
        void computeTSNE(std::variant<const SparseMatHDI*, const utils::GraphBaseInterface*> input);

        void computeUMAP(std::variant<const SparseMatHDI*, const utils::GraphBaseInterface*> input);

    public: // Setter
        void setSettings(const ComputeEmbeddingSettings& s) { _settings = s; };
        void setSettingsTsne(const TsneEmbeddingParameters& tsne) { _settings.tsne = tsne; };
        void setSettingsUmap(const UmapEmbeddingParameters& umap) { _settings.umap = umap; };

    public: // Getter
        ComputeEmbeddingSettings& getSettings() { return _settings; };
        const ComputeEmbeddingSettings& getSettings() const { return _settings; };

        auto& getInitEmbedding() { return _initEmbedding; };
        const auto& getInitEmbedding() const { return _initEmbedding; };

        auto getEmbedding() { return _currentEmbedding; }
        const auto& getEmbedding() const { return _currentEmbedding; }

    private:
        void runTSNE(TsneComputation& tsne);
        void runUMAP(UmapComputation& umap);

    private:

        // Data
        std::vector<float> _initEmbedding = {};
        std::vector<float> _currentEmbedding = {};

        // Settings
        ComputeEmbeddingSettings _settings = {};

    };


} // namespace sph
