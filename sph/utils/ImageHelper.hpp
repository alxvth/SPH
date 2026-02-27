#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "Settings.hpp"

namespace sph::utils
{
    struct ConnectedNeighbors {

        constexpr static int nn_eight = 8;
        constexpr static std::array<int, 8> rd_eight = { -1, -1, -1,  0, 0,  1, 1, 1 };
        constexpr static std::array<int, 8> cd_eight = { -1,  0,  1, -1, 1, -1, 0, 1 };

        constexpr static int nn_four = 4;
        constexpr static std::array<int, 4> rd_four = { -1, 0, 1,  0 };
        constexpr static std::array<int, 4> cd_four = { 0, 1, 0, -1 };

        int num_n() const {
            switch (neighborConnection)
            {
            case NeighConnection::EIGHT: return nn_eight;
            case NeighConnection::FOUR:  return nn_four;
            }

            return nn_four;
        }

        int row_dir(int i) const {
            switch (neighborConnection)
            {
            case NeighConnection::EIGHT: return rd_eight[i];
            case NeighConnection::FOUR:  return rd_four[i];
            }

            return rd_four[i];
        }

        int col_dir(int i) const {
            switch (neighborConnection)
            {
            case NeighConnection::EIGHT: return cd_eight[i];
            case NeighConnection::FOUR:  return cd_four[i];
            }

            return cd_four[i];
        }

        NeighConnection neighborConnection = NeighConnection::FOUR;
    };


    struct ImageInfo {
        uint64_t numCols { 0 };
        uint64_t numRows { 0 };
        ConnectedNeighbors neighConnection = {};
    };

    // Returns the IDs of the four or eight connected neighbors (depending on neighConnection)
    void pixelNeighborIDs(uint64_t numCols, uint64_t numRows, const ConnectedNeighbors& neighConnection, uint64_t id, std::vector<uint64_t>& neighIDs);

}
