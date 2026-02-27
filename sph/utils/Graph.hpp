#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace sph::utils {

    /// ////////////////////// ///
    /// Graph Base & Interface ///
    /// ////////////////////// ///

    // These Graph structures assume that each point/node has at least one neighbor
    // These Graph structures assume the first neighbor of a point/node to be the point itself

    struct GraphBaseInterface
    {
        virtual inline std::span<const int64_t> getNeighbors(uint64_t id) const = 0;

        virtual inline std::span<const float> getDistances(uint64_t id) const = 0;

        virtual inline std::span<float> getDistancesRef(uint64_t id) = 0;

        virtual inline std::span<int64_t> getNeighborsRef(uint64_t id) = 0;

        virtual inline int64_t getNeighborN(uint64_t id, uint64_t n) const = 0;

        // n is the nth neighbor of id
        virtual inline float getDistanceN(uint64_t id, uint64_t n) const = 0;

        // returns infinity if id2 is not a neighbor of id1
        virtual inline float getDistance(uint64_t id1, uint64_t id2) const = 0;

        virtual inline std::pair<int64_t, float> getNeighborDistanceN(uint64_t id, uint64_t n) const = 0;

        virtual inline int64_t getK([[maybe_unused]] uint64_t id = 0) const = 0;

        // returns the neighbor position in the neighbor span if n is a neighbor of id, otherwise -1
        virtual int64_t isDirectNeighbor(uint64_t id1, uint64_t id2) const = 0;

        virtual inline int64_t getNumPoints() const = 0;
        virtual const std::vector<int64_t>& getKnnIndices() const = 0;
        virtual std::vector<int64_t>& getKnnIndices() = 0;
        virtual const std::vector<float>& getKnnDistances() const = 0;
        virtual std::vector<float>& getKnnDistances() = 0;

        virtual inline bool isValid() const = 0;

        virtual inline bool isSymmetric() const = 0;

        virtual inline void clear() = 0;

        struct Edge {
            int64_t source = {};
            int64_t target = {};
            float weight = {};
        };

        inline size_t getNumEdges() const { return getKnnDistances().size() - getNumPoints(); }

        bool hasUniqueNeighbors() const;

        virtual ~GraphBaseInterface() = default;
    };

    struct GraphBaseData
    {
        GraphBaseData() = default;
        GraphBaseData(const GraphBaseData& graphBase);
        GraphBaseData(GraphBaseData&& graphBase) noexcept;
        GraphBaseData(std::vector<int64_t>&& idx, std::vector<float>&& dists, int64_t numPoints, bool symmetric) noexcept;

        std::vector<int64_t>    knnIndices = {};      // indices of nearest neighbors, sorted by distance
        std::vector<float>      knnDistances = {};    // distances to nearest neighbors, sorted
        int64_t                 numPoints = 0;       // number of points/vertices

        bool                    symmetric = false;

        virtual inline bool isValid() const {
            return (knnIndices.size() > 0) &&
                (knnDistances.size() > 0) &&
                (numPoints >= 0) &&
                (knnIndices.size() == knnDistances.size());
        }

        virtual inline void clear() {
            knnIndices.clear();
            knnDistances.clear();
            numPoints = 0;
            symmetric = false;
        }

        inline bool operator==(const GraphBaseData& other) const {
            return (knnIndices == other.knnIndices) && (knnDistances == other.knnDistances) && (numPoints == other.numPoints);
        }

    };

    struct GraphBaseView
    {
        GraphBaseView() = default;
        GraphBaseView(const GraphBaseView& graphBase);
        GraphBaseView(std::vector<int64_t>* idx, std::vector<float>* dists, int64_t* numPoints, bool* symmetric);

        std::vector<int64_t>*   knnIndices = nullptr;      // indices of nearest neighbors, sorted by distance
        std::vector<float>*     knnDistances = nullptr;    // distances to nearest neighbors, sorted
        int64_t*                numPoints = nullptr;       // number of points/vertices

        bool*                   symmetric = nullptr;

        virtual inline bool isValid() const {
            return (knnIndices != nullptr) &&
                (knnDistances != nullptr) &&
                (numPoints != nullptr) &&
                (knnIndices->size() > 0) &&
                (knnDistances->size() > 0) &&
                (*numPoints >= 0);
        }

        inline void clear() {
            knnIndices->clear();
            knnDistances->clear();
            *numPoints = 0;
            *symmetric = false;
        }

        inline bool operator==(const GraphBaseView& other) const {
            return (knnIndices == other.knnIndices) && (knnDistances == other.knnDistances) && (*numPoints == *other.numPoints);
        }

    };

    /// //////////// ///
    /// Graph & View ///
    /// //////////// ///

    struct GraphInterface : public GraphBaseInterface
    {
        virtual const std::vector<int64_t>& getNns() const = 0;
        virtual std::vector<int64_t>& getNns() = 0;
        virtual const std::vector<int64_t>& getOffsets() const = 0;
        virtual std::vector<int64_t>& getOffsets() = 0;

        virtual void updateOffsets() = 0;
        virtual void updateFixedNumNeighbors(int64_t k) = 0;

    };

    struct GraphView;

    struct Graph : public GraphBaseData, public GraphInterface
    {
        Graph() = default;
        Graph(const Graph& graph);
        Graph& operator= (const Graph& graph);
        Graph(Graph&& graphBase) noexcept;
        Graph(std::vector<int64_t>&& idx, std::vector<float>&& dists, std::vector<int64_t>&& numNeighbors, std::vector<int64_t>&& offset, bool symmetric) noexcept;

        std::vector<int64_t>     nns = {};       // number of neighbors per point
        std::vector<int64_t>     offsets = {};   // helper for indexing

        inline int64_t getK([[maybe_unused]] uint64_t id = 0) const override { return nns[id]; }

        const std::vector<int64_t>& getNns() const override { return nns; }
        std::vector<int64_t>& getNns() override { return nns; }

        const std::vector<int64_t>& getOffsets() const override { return offsets; }
        std::vector<int64_t>& getOffsets() override { return offsets; }

        inline int64_t getNumPoints() const override { return numPoints; }
        const std::vector<int64_t>& getKnnIndices() const override { return knnIndices; };
        std::vector<int64_t>& getKnnIndices() override { return knnIndices; };
        const std::vector<float>& getKnnDistances() const override { return knnDistances; };
        std::vector<float>& getKnnDistances() override { return knnDistances; };

        inline std::span<const int64_t> getNeighbors(uint64_t id) const override {
            const auto nn = nns[id];
            const auto skip = offsets[id];
            return std::span{knnIndices.begin() + skip, knnIndices.begin() + skip + nn };
        };

        inline std::span<const float> getDistances(uint64_t id) const override {
            const auto nn = nns[id];
            const auto skip = offsets[id];
            return std::span{knnDistances.begin() + skip, knnDistances.begin() + skip + nn};
        };

        inline std::span<float> getDistancesRef(uint64_t id) override {
            const auto nn = nns[id];
            const auto skip = offsets[id];
            return std::span{knnDistances.begin() + skip, knnDistances.begin() + skip + nn};
        };

        inline std::span<int64_t> getNeighborsRef(uint64_t id) override {
            const auto nn = nns[id];
            const auto skip = offsets[id];
            return std::span{ knnIndices.begin() + skip, knnIndices.begin() + skip + nn };
        };

        inline int64_t getNeighborN(uint64_t id, uint64_t n) const override {
            const auto skip = offsets[id];
            assert(static_cast<int64_t>(n) <= nns[id]);
            return knnIndices[skip + n];
        }

        inline float getDistanceN(uint64_t id, uint64_t n) const override {
            const auto skip = offsets[id];
            assert(static_cast<int64_t>(n) <= nns[id]);
            return knnDistances[skip + n];
        }

        inline float getDistance(uint64_t id1, uint64_t id2) const override {
            const auto skip = offsets[id1];
            const auto n = isDirectNeighbor(id1, id2);

            if(n == -1)
                return  std::numeric_limits<float>::infinity();

            return knnDistances[skip + n];
        }

        inline std::pair<int64_t, float> getNeighborDistanceN(uint64_t id, uint64_t n) const override {
            const auto skip = offsets[id];
            return { knnIndices[skip + n], knnDistances[skip + n] };
        }

        int64_t isDirectNeighbor(uint64_t id1, uint64_t id2) const override;

        GraphView getGraphView();
        const GraphView getGraphView() const;

        inline bool isSymmetric() const override { 
            return symmetric; 
        };

        inline bool isValid() const override {
            return GraphBaseData::isValid() &&
                (nns.size() == static_cast<size_t>(numPoints)) && 
                (offsets.size() == static_cast<size_t>(numPoints)) &&
                (static_cast<size_t>(offsets.back() + nns.back()) == knnIndices.size()) &&
                (static_cast<size_t>(std::accumulate(nns.begin(), nns.end(), 0ll)) == knnIndices.size());
        }

        inline void clear() override {
            GraphBaseData::clear();
            nns.clear();
            offsets.clear();
        }

        void updateOffsets() override
        {
            offsets.resize(nns.size());
            std::exclusive_scan(nns.cbegin(), nns.cend(), offsets.begin(), 0ll);
        }

        void updateFixedNumNeighbors(int64_t k) override
        {
            nns.clear();
            nns.resize(numPoints, k);
            updateOffsets();
        }

        inline bool operator==(const Graph& other) const {
            if (*this == static_cast<const GraphBaseData&>(other)) {
                return nns == other.nns && offsets == other.offsets;;
            }
            return false;
        }
    };

    struct GraphView : public GraphBaseView, public GraphInterface
    {
        GraphView() = default;
        GraphView(const GraphView& graphView);
        GraphView& operator= (const GraphView& graphView);
        GraphView(std::vector<int64_t>* idx, std::vector<float>* dists, int64_t* numPoints, std::vector<int64_t>* ns, std::vector<int64_t>* offset, bool* symmetric);

        std::vector<int64_t>* nns = nullptr;        // number of neighbors per point
        std::vector<int64_t>* offsets = nullptr;    // helper for indexing

        inline int64_t getK([[maybe_unused]] uint64_t id = 0) const override { return (*nns)[id]; }

        const std::vector<int64_t>& getNns() const override { return *nns; }
        std::vector<int64_t>& getNns() override { return *const_cast<std::vector<int64_t>*>(nns); }

        const std::vector<int64_t>& getOffsets() const override { return *offsets; }
        std::vector<int64_t>& getOffsets() override { return *const_cast<std::vector<int64_t>*>(offsets); }

        inline int64_t getNumPoints() const override { return *numPoints; }
        const std::vector<int64_t>& getKnnIndices() const override { return *knnIndices; };
        std::vector<int64_t>& getKnnIndices() override { return *knnIndices; };
        const std::vector<float>& getKnnDistances() const override { return *knnDistances; };
        std::vector<float>& getKnnDistances() override { return *knnDistances; };

        inline std::span<const int64_t> getNeighbors(uint64_t id) const override {
            const auto nn = (*nns)[id];
            const auto skip = (*offsets)[id];
            return std::span{ knnIndices->data() + skip, knnIndices->data() + skip + nn };
        };

        inline std::span<const float> getDistances(uint64_t id) const override {
            const auto nn = (*nns)[id];
            const auto skip = (*offsets)[id];
            return std::span{ knnDistances->data() + skip, knnDistances->data() + skip + nn };
        };

        inline std::span<int64_t> getNeighborsRef(uint64_t id) override {
            const auto nn = (*nns)[id];
            const auto skip = (*offsets)[id];
            return std::span{ knnIndices->data() + skip, knnIndices->data() + skip + nn };
        };

        inline std::span<float> getDistancesRef(uint64_t id) override {
            const auto nn = (*nns)[id];
            const auto skip = (*offsets)[id];
            return std::span{ knnDistances->data() + skip, knnDistances->data() + skip + nn };
        };

        inline int64_t getNeighborN(uint64_t id, uint64_t n) const override {
            const auto skip = (*offsets)[id];
            assert(static_cast<int64_t>(n) <= (*nns)[id]);
            return (*knnIndices)[skip + n];
        }

        inline float getDistanceN(uint64_t id, uint64_t n) const override {
            const auto skip = (*offsets)[id];
            assert(static_cast<int64_t>(n) <= (*nns)[id]);
            return (*knnDistances)[skip + n];
        }

        inline float getDistance(uint64_t id1, uint64_t id2) const override {
            const auto skip = (*offsets)[id1];
            const auto n = isDirectNeighbor(id1, id2);

            if (n == -1)
                return  std::numeric_limits<float>::infinity();

            return (*knnDistances)[skip + n];
        }

        inline std::pair<int64_t, float> getNeighborDistanceN(uint64_t id, uint64_t n) const override {
            const auto skip = (*offsets)[id];
            return { (*knnIndices)[skip + n], (*knnDistances)[skip + n] };
        }

        int64_t isDirectNeighbor(uint64_t id1, uint64_t id2) const override;

        inline bool isSymmetric() const override { 
            return symmetric ? *symmetric : false; 
        };

        inline bool isValid() const override {
            return GraphBaseView::isValid() &&
                (nns != nullptr) && 
                (nns->size() == static_cast<size_t>(*numPoints)) && 
                (offsets != nullptr) &&
                (offsets->size() == static_cast<size_t>(*numPoints)) &&
                (static_cast<size_t>(offsets->back() + nns->back()) == knnIndices->size()) &&
                (static_cast<size_t>(std::accumulate(nns->begin(), nns->end(), 0ll)) == knnIndices->size());
        }

        inline void clear() override {
            GraphBaseView::clear();
            nns->clear();
            offsets->clear();
        }

        void updateOffsets() override
        {
            offsets->resize(nns->size());
            std::exclusive_scan(nns->cbegin(), nns->cend(), offsets->begin(), 0ll);
        }

        void updateFixedNumNeighbors(int64_t k) override
        {
            nns->clear();
            nns->resize(*numPoints, k);
            updateOffsets();
        }

        inline bool operator==(const GraphView& other) const {
            if (*this == static_cast<const GraphBaseView&>(other)) {
                return nns == other.nns && offsets == other.offsets;
            }
            return false;
        }
    };

    /// ///////////// ///
    /// KGraph & View ///
    /// ///////////// ///

    // KGraph are never symmetric

    struct KGraphView;
        
    struct KGraph : public GraphBaseData, public GraphBaseInterface
    //struct [[deprecated("KGraph is deprecated and might be removed. Use Graph. No Boost overloads exist for KGraph.")]] KGraph : public GraphBaseData, public GraphBaseInterface
    {
        KGraph() = default;
        KGraph(const KGraph& kGraph);
        KGraph& operator= (const KGraph& kGraph);
        KGraph(KGraph&& kGraph) noexcept;
        KGraph(std::vector<int64_t>&& idx, std::vector<float>&& dists, int64_t k) noexcept;

        int64_t k = 0; // number of neighbors

        inline int64_t getK([[maybe_unused]] uint64_t id = 0) const override { return k; }

        inline int64_t getNumPoints() const override { return numPoints; }
        const std::vector<int64_t>& getKnnIndices() const override { return knnIndices; };
        std::vector<int64_t>& getKnnIndices() override { return knnIndices; };
        const std::vector<float>& getKnnDistances() const override { return knnDistances; };
        std::vector<float>& getKnnDistances() override { return knnDistances; };

        inline std::span<const int64_t> getNeighbors(uint64_t id) const override {
            return std::span{ knnIndices.data() + id * k, knnIndices.data() + id * k + k };
        };

        inline std::span<const float> getDistances(uint64_t id) const override {
            return std::span{ knnDistances.data() + id * k, knnDistances.data() + id * k + k };
        };

        inline std::span<int64_t> getNeighborsRef(uint64_t id) override {
            return std::span{ knnIndices.data() + id * k, knnIndices.data() + id * k + k };
        };

        inline std::span<float> getDistancesRef(uint64_t id) override {
            return std::span{ knnDistances.data() + id * k, knnDistances.data() + id * k + k };
        };

        inline int64_t getNeighborN(uint64_t id, uint64_t n) const override {
            return knnIndices[id * k + n];
        }

        inline float getDistanceN(uint64_t id, uint64_t n) const override {
            return knnDistances[id * k + n];
        }

        inline float getDistance(uint64_t id1, uint64_t id2) const override {
            const auto n = isDirectNeighbor(id1, id2);

            if (n == -1)
                return  std::numeric_limits<float>::infinity();

            return knnDistances[id1 * k + n];
        }

        inline std::pair<int64_t, float> getNeighborDistanceN(uint64_t id, uint64_t n) const override {
            return { knnIndices[id * k + n], knnDistances[id * k + n] };
        }

        int64_t isDirectNeighbor(uint64_t id1, uint64_t id2) const override;

        KGraphView getKGraphView();
        const KGraphView getKGraphView() const;

        inline bool isSymmetric() const override { 
            return symmetric; 
        };

        inline bool isValid() const override {
            return GraphBaseData::isValid() &&
                (k >= 0) && 
                (static_cast<size_t>(k * numPoints) == knnIndices.size());
        }

        inline void clear() override {
            GraphBaseData::clear();
            k = 0;
        }

        inline bool operator==(const KGraph& other) const {
            if (*this == static_cast<const GraphBaseData&>(other)) {
                return k == other.k;
            }
            return false;
        }
    };

    struct KGraphView : public GraphBaseView, public GraphBaseInterface
    //struct [[deprecated("KGraphView is deprecated and might be removed. Use GraphView. No Boost overloads exist for KGraphView.")]] KGraphView : public GraphBaseView, public GraphBaseInterface
    {
        KGraphView() = default;
        KGraphView(const KGraphView& kGraphView);
        KGraphView& operator= (const KGraphView& kGraphView);
        KGraphView(std::vector<int64_t>* idx, std::vector<float>* dists, int64_t* numPoints, int64_t* k);

        int64_t* k = nullptr; // number of neighbors

        inline int64_t getK([[maybe_unused]] uint64_t id = 0) const override { return *k; }

        inline int64_t getNumPoints() const override { return *numPoints; }
        const std::vector<int64_t>& getKnnIndices() const override { return *knnIndices; };
        std::vector<int64_t>& getKnnIndices() override { return *knnIndices; };
        const std::vector<float>& getKnnDistances() const override { return *knnDistances; };
        std::vector<float>& getKnnDistances() override { return *knnDistances; };

        inline std::span<const int64_t> getNeighbors(uint64_t id) const override {
            return std::span{ knnIndices->data() + id * *k, knnIndices->data() + id * *k + *k };
        };

        inline std::span<const float> getDistances(uint64_t id) const override {
            return std::span{ knnDistances->data() + id * *k, knnDistances->data() + id * *k + *k };
        };

        inline std::span<int64_t> getNeighborsRef(uint64_t id) override {
            return std::span{ knnIndices->data() + id * *k, knnIndices->data() + id * *k + *k };
        };

        inline std::span<float> getDistancesRef(uint64_t id) override {
            return std::span{ knnDistances->data() + id * *k, knnDistances->data() + id * *k + *k };
        };

        inline int64_t getNeighborN(uint64_t id, uint64_t n) const override {
            return (*knnIndices)[id * *k + n];
        }

        inline float getDistanceN(uint64_t id, uint64_t n) const override {
            return (*knnDistances)[id * *k + n];
        }

        inline float getDistance(uint64_t id1, uint64_t id2) const override {
            const auto n = isDirectNeighbor(id1, id2);

            if (n == -1)
                return  std::numeric_limits<float>::infinity();

            return (*knnDistances)[id1 * *k + n];
        }

        inline std::pair<int64_t, float> getNeighborDistanceN(uint64_t id, uint64_t n) const override {
            return { (*knnIndices)[id * *k + n], (*knnDistances)[id * *k + n] };
        }

        int64_t isDirectNeighbor(uint64_t id1, uint64_t id2) const override;

        virtual inline bool isSymmetric() const override { 
            return symmetric ? *symmetric : false; 
        };

        virtual inline bool isValid() const override {
            return GraphBaseView::isValid() &&
                (k != nullptr) && 
                (*k >= 0) && 
                (static_cast<size_t>(*k * *numPoints) == knnIndices->size());
        }

        virtual inline void clear() override {
            GraphBaseView::clear();
            *k = 0;
        }

        inline bool operator==(const KGraphView& other) const {
            if (*this == static_cast<const GraphBaseView&>(other)) {
                return k == other.k;
            }
            return false;
        }
    };


    /// ///////// ///
    /// Utilities ///
    /// ///////// ///

    // Append a new vertex and init the knn neighbor IDs
    void appendNode(Graph& g, std::vector<uint64_t>&& newNeighborIDs);

    // Append a new vertex and init the knn neighbor IDs and distances
    void appendNode(Graph& g, std::vector<uint64_t>&& newNeighborIDs, std::vector<float>&& newNeighborDistances);

    // TODO: insert edge(s) 
    // affect e.g. ImageHierarchy (Graph similarities)

} // namespace sph::utils
