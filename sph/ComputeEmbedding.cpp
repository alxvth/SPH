#include "ComputeEmbedding.hpp"

#include "EmbedUmap.hpp"
#include "EmbedTsne.hpp"

#include <sph/utils/CommonDefinitions.hpp>
#include <sph/utils/Logger.hpp>
#include <sph/utils/Math.hpp>
#include <sph/utils/ProgressBar.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace sph {

    static void resizeEmbedding(std::vector<float>& v, uint64_t numEmbPoints)
    {
        v.clear();
        v.resize(numEmbPoints * 2, 0);
    }

    void ComputeEmbedding::initEmbedding(uint64_t numEmbPoints)
    {
        resizeEmbedding(_initEmbedding, numEmbPoints);

        randomEmbeddingInit(_initEmbedding, _settings.initRadius, _settings.initRadius);
    }

    void ComputeEmbedding::initEmbedding([[maybe_unused]] uint64_t numEmbPoints, std::vector<float>&& embedding)
    {
        assert(embedding.size() == numEmbPoints * 2);

        _initEmbedding = std::move(embedding);
    }

    void ComputeEmbedding::randomEmbeddingInit(std::vector<float>& emb, float x, float y) {
        const int64_t numEmbPoints = emb.size() / 2;

        SPH_PARALLEL
        for (int64_t i = 0; i < numEmbPoints; i++) {
            const auto randomPoint = utils::randomVec(x, y);

            emb[2ll * i] = randomPoint.first;
            emb[2ll * i + 1] = randomPoint.second;
        }

    }

    void ComputeEmbedding::computeTSNE(std::variant<const SparseMatHDI*, const utils::GraphBaseInterface*> input)
    {
        int64_t numPoints = 0;
        TsneComputation tsneComputation;

        if (std::holds_alternative<const utils::GraphBaseInterface*>(input)) {
            const auto graph = std::get<const utils::GraphBaseInterface*>(input);
            tsneComputation.setNeighborGraph(graph);
            numPoints = graph->getNumPoints();
        }

        if (std::holds_alternative<const SparseMatHDI*>(input)) {
            const auto sparseMat = std::get<const SparseMatHDI*>(input);
            tsneComputation.setProbabilityDistribution(sparseMat);
            numPoints = sparseMat->size();
        }

        if (numPoints == 1) {
            Log::info("ComputeEmbedding:computeTSNE: Only 1 point, do not embed.");
            _initEmbedding = { 0.f, 0.f };
            _currentEmbedding = { 0.f, 0.f };
            return;
        }

        if (_initEmbedding.empty()) {
            initEmbedding(numPoints);
        }

        runTSNE(tsneComputation);

        utils::Logger::flush();
    }

    void ComputeEmbedding::runTSNE(TsneComputation& tsne)
    {
        Log::info("ComputeEmbedding:: compute TSNE...");

        tsne.setParams(_settings.tsne);
        tsne.setInitialEmbedding(_initEmbedding);    // updates params.gradDescentParams._presetEmbedding, i.e. call after setParams()

#ifdef __RUNTIME_GPU__
        auto offscreenBuffer = std::make_unique<OffscreenBufferGLFW>();
        tsne.setOffscreenBuffer(offscreenBuffer.get());
#endif

        const uint32_t totalIterations = _settings.tsne.numIterations;

        if (totalIterations == 0)
            return;

        const uint32_t updateStep = 10;
        uint32_t currentIteration = std::min(updateStep, totalIterations);

        utils::ProgressBar progress(totalIterations);
        tsne.compute(currentIteration, false);
        progress.update(currentIteration);

        if (totalIterations > currentIteration) {
            while (currentIteration + updateStep <= totalIterations) {
                currentIteration += updateStep;
                tsne.continueGradientDescent(updateStep, false);
                progress.update(currentIteration);
            }

            // Handle the remainder if there's any
            if (currentIteration < totalIterations) {
                tsne.continueGradientDescent(totalIterations - currentIteration, false);
            }
        }
        progress.finish();

#ifdef __RUNTIME_GPU__
        offscreenBuffer->destroyContext();
        tsne.setOffscreenBuffer(nullptr);
#endif

        _currentEmbedding = tsne.getEmbedding().getContainer();
    }

    void ComputeEmbedding::computeUMAP(std::variant<const SparseMatHDI*, const utils::GraphBaseInterface*> input)
    {
        UmapComputation umap;

        int64_t numPoints = 0;

        if (std::holds_alternative<const utils::GraphBaseInterface*>(input)) {
            const auto graph = std::get<const utils::GraphBaseInterface*>(input);
            umap.setNeighborGraph(graph);
            numPoints = graph->getNumPoints();
        }

        if (std::holds_alternative<const SparseMatHDI*>(input)) {
            const auto sparseMat = std::get<const SparseMatHDI*>(input);
            umap.setNeighborMatrix(sparseMat);
            numPoints = sparseMat->size();
        }

        if (numPoints == 1) {
            Log::info("ComputeEmbedding:computeTSNE: Only 1 point, do not embed.");
            _initEmbedding = { 0.f, 0.f };
            _currentEmbedding = { 0.f, 0.f };
            return;
        }

        if (_initEmbedding.empty()) {
            initEmbedding(numPoints);
        }

        runUMAP(umap);

        utils::Logger::flush();
    }

    void ComputeEmbedding::runUMAP(UmapComputation& umap)
    {
        Log::info("ComputeEmbedding:: compute UMAP...");

        umap.setParams(_settings.umap);
        umap.setInitialEmbedding(_initEmbedding);

        umap.compute();
        _currentEmbedding = umap.getEmbedding();
    }

} // namespace sph

