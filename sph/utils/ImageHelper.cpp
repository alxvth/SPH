#include "ImageHelper.hpp"

#include <cassert>

namespace sph::utils
{

    void pixelNeighborIDs(uint64_t numCols, uint64_t numRows, const ConnectedNeighbors& neighConnection, uint64_t id, std::vector<uint64_t>& neighIDs)
    {
        assert(id < numCols * numRows);

        neighIDs.clear();
        neighIDs.reserve(neighConnection.num_n());

        const uint64_t idRow = id / numCols;
        const uint64_t idCol = id % numCols;

        for (int i = 0; i < neighConnection.num_n(); i++)
        {
            auto newRow = idRow + neighConnection.row_dir(i);
            auto newCol = idCol + neighConnection.col_dir(i);

            // Check if the neighbor is within the image boundaries
            if (newRow < numRows && newCol < numCols) {
                neighIDs.push_back(newRow * numCols + newCol);;
            }
        }
    }

}
