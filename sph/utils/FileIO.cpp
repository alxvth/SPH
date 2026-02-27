#include "FileIO.hpp"

#include "CommonDefinitions.hpp"
#include "Graph.hpp"
#include "Logger.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/SparseCore>
#include <hdi/data/map_mem_eff.h>
#include <lz4.h>

namespace sph::utils {

    void __logMessage__(const std::string& msg)
    {
        Log::info("{}", msg);
    }

    void __debugMessage__(const std::string& msg)
    {
        Log::debug("{}", msg);
    }

    void __errorMessage__(const std::string& msg)
    {
        Log::error("Caching failed: {}", msg);
    }

    void __errorMessageLoading__()
    {
        Log::error("Caching failed. File could not be opened.");
    }

    // Helper function to calculate total bytes needed for nested vectors of sparse vectors
    static std::pair<size_t, bool> calculateTotalBytesSparseMatSPH(const SparseMatSPH& vec, bool verbose = true) {
        size_t totalBytes = sizeof(size_t); // Vec size
        for (const auto& sparseVec : vec) {
            totalBytes += sizeof(Eigen::Index) * 2; // rows and nonZeros
            totalBytes += sparseVec.nonZeros() * (sizeof(SparseVecSPH::StorageIndex) + sizeof(float)); // indices and values
        }

        bool fitsInSingle = true;

        if (totalBytes > std::numeric_limits<int>::max())
        {
            if(verbose)
                Log::warn("calculateTotalBytesSparseMatSPH:: File too large for default lz4 compression.");

            fitsInSingle = false;
        }

        return { totalBytes, fitsInSingle };
    }

    // Helper function to calculate total bytes needed for nested vectors of sparse vectors
    static std::pair<size_t, bool> calculateTotalBytesSparseMatSPH(SparseMatSPH::const_iterator start, SparseMatSPH::const_iterator end) {
        size_t totalBytes = sizeof(size_t); // Vec size
        for (auto& it = start; it != end; it++) {
            totalBytes += sizeof(Eigen::Index) * 2; // rows and nonZeros
            totalBytes += it->nonZeros() * (sizeof(SparseVecSPH::StorageIndex) + sizeof(float)); // indices and values
        }

        bool fitsInSingle = true;

        if (totalBytes > std::numeric_limits<int>::max())
        {
            Log::warn("calculateTotalBytesSparseMatSPH:: File too large for default lz4 compression.");
            fitsInSingle = false;
        }

        return { totalBytes, fitsInSingle };
    }

    // Helper function to calculate total bytes needed for nested vectors of sparse vectors
    static size_t calculateTotalBytesVecOfSparseMatSPH(const std::vector<SparseMatSPH>& vec) {
        size_t totalBytes = sizeof(size_t); // Outermost size
        for (const auto& middleVec : vec) {
            totalBytes += sizeof(size_t); // Middle size
            for (const auto& sparseVec : middleVec) {
                totalBytes += sizeof(Eigen::Index) * 2; // rows and nonZeros
                totalBytes += sparseVec.nonZeros() * (sizeof(SparseVecSPH::StorageIndex) + sizeof(float)); // indices and values
            }
        }

        if (totalBytes > std::numeric_limits<int>::max())
            Log::error("calculateTotalBytesVecOfSparseMatSPH:: File too large for default lz4 compression. Proceeding, but expect errors and faulty files.");

        return totalBytes;
    }

    // Helper function to calculate total bytes needed for nested vectors of pairs
    static std::pair<size_t, bool> calculateTotalBytesSparseMatHDI(const SparseMatHDI& vec, bool verbose = true) {
        size_t totalBytes = sizeof(size_t); // Outer size
        for (const auto& innerVec : vec) {
            totalBytes += sizeof(size_t); // Inner size
            totalBytes += innerVec.size() * (sizeof(uint32_t) + sizeof(float)); // Pairs data
        }
        
        bool fitsInSingle = true;

        if (totalBytes > std::numeric_limits<int>::max())
        {
            if (verbose)
                __errorMessage__("calculateTotalBytesSparseMatHDI:: File too large for default lz4 compression. Proceeding, but expect errors and faulty files.");

            fitsInSingle = false;
        }

        return { totalBytes, fitsInSingle };
    }

    static std::pair<size_t, bool> calculateTotalBytesSparseMatHDI(SparseMatHDI::const_iterator start, SparseMatHDI::const_iterator end) {
        size_t totalBytes = sizeof(size_t); // Outer size
        for (auto& it = start; it != end; it++) {
            totalBytes += sizeof(size_t); // Inner size
            totalBytes += it->size() * (sizeof(uint32_t) + sizeof(float)); // Pairs data
        }

        bool fitsInSingle = true;

        if (totalBytes > std::numeric_limits<int>::max())
        {
            Log::warn("calculateTotalBytesSparseMatHDI:: File too large for default lz4 compression.");
            fitsInSingle = false;
        }

        return { totalBytes, fitsInSingle };
    }

    std::vector<std::span<size_t>> divide_into_spans(std::vector<size_t>& data, size_t n) {
        std::vector<std::span<size_t>> spans;
        if (n == 0 || data.empty()) return spans;

        size_t base_size = data.size() / n;
        size_t remainder = data.size() % n;
        size_t offset = 0;

        for (size_t i = 0; i < n; ++i) {
            size_t chunk_size = base_size + (i < remainder ? 1 : 0);  // Extra element for the first 'remainder' spans
            spans.emplace_back(data.data() + offset, chunk_size);
            offset += chunk_size;
        }

        return spans;
    }

    int lz4_maxCompressedSize(int totalBytes)
    {
        return LZ4_compressBound(totalBytes);
    }

    int lz4_compress(const char* src, char* dst, int srcSize, int dstCapacity)
    {
        return LZ4_compress_default(src, dst, srcSize, dstCapacity);
    }

    int lz4_decompress(const char* src, char* dst, int compressedSize, int dstCapacity)
    {
        return LZ4_decompress_safe(src, dst, compressedSize, dstCapacity);
    }

    /// /////// ///
    /// WRITING ///
    /// /////// ///

    bool writeSparseMatHDIToBinary(const std::string& writePath, const SparseMatHDI& vecOfSparseMatSPH)
    {
        std::ofstream fout(writePath, std::ios::binary);
        if (!fileOpens(fout)) return false;

        // Write the size of the outermost vector
        std::size_t outermostSize = vecOfSparseMatSPH.size();
        fout.write(reinterpret_cast<const char*>(&outermostSize), sizeof(std::size_t));

        // Write vectors
        for (const auto& innerVec : vecOfSparseMatSPH) {
            // Write inner size
            const size_t innerSize = innerVec.size();
            fout.write(reinterpret_cast<const char*>(&innerSize), sizeof(std::size_t));

            // Write pairs
            for (const auto& pair : innerVec) {
                fout.write(reinterpret_cast<const char*>(&pair.first), sizeof(uint32_t));
                fout.write(reinterpret_cast<const char*>(&pair.second), sizeof(float));
            }
        }

        fout.close();
        return true;
    }

    bool writeSparseMatSPHToBinary(const std::string& writePath, const SparseMatSPH& vecOfSparseMatSPH)
    {
        std::ofstream fout(writePath, std::ios::binary);
        if (!fileOpens(fout)) return false;

        // Write the size of the outermost vector
        std::size_t outermostSize = vecOfSparseMatSPH.size();
        fout.write(reinterpret_cast<const char*>(&outermostSize), sizeof(std::size_t));

        // Write each sparse vector
        for (const auto& sparseVector : vecOfSparseMatSPH) {

            // Write the data of the SparseVector
            Eigen::Index rows = sparseVector.rows();
            Eigen::Index nnzs = sparseVector.nonZeros();

            fout.write(reinterpret_cast<const char*>(&rows), sizeof(Eigen::Index));
            fout.write(reinterpret_cast<const char*>(&nnzs), sizeof(Eigen::Index));

            // Loop over the non-zero elements and write their indices and values
            for (SparseVecSPH::InnerIterator it(sparseVector); it; ++it) {
                SparseVecSPH::StorageIndex index = it.index();
                float value = it.value();
                fout.write(reinterpret_cast<const char*>(&index), sizeof(SparseVecSPH::StorageIndex));
                fout.write(reinterpret_cast<const char*>(&value), sizeof(float));
            }

        }

        fout.close();
        return true;
    }

    bool writeVecOfSparseMatSPHToBinary(const std::string& writePath, const std::vector<SparseMatSPH>& vecOfSparseMatSPH)
    {
        std::ofstream fout(writePath, std::ios::binary);
        if (!fileOpens(fout)) return false;

        // Write the size of the outermost vector
        std::size_t outermostSize = vecOfSparseMatSPH.size();
        fout.write(reinterpret_cast<const char*>(&outermostSize), sizeof(std::size_t));

        // Write each middle vector
        for (const auto& middleVector : vecOfSparseMatSPH) {
            // Write the size of the middle vector
            std::size_t middleSize = middleVector.size();
            fout.write(reinterpret_cast<const char*>(&middleSize), sizeof(std::size_t));

            // Write each inner (sparse) vector
            for (const auto& sparseVector : middleVector) {

                // Write the data of the SparseVector
                Eigen::Index rows = sparseVector.rows();
                Eigen::Index nnzs = sparseVector.nonZeros();

                fout.write(reinterpret_cast<const char*>(&rows), sizeof(Eigen::Index));
                fout.write(reinterpret_cast<const char*>(&nnzs), sizeof(Eigen::Index));

                // Loop over the non-zero elements and write their indices and values
                for (SparseVecSPH::InnerIterator it(sparseVector); it; ++it) {
                    SparseVecSPH::StorageIndex index = it.index();
                    float value = it.value();
                    fout.write(reinterpret_cast<const char*>(&index), sizeof(SparseVecSPH::StorageIndex));
                    fout.write(reinterpret_cast<const char*>(&value), sizeof(float));
                }

            }
        }

        fout.close();
        return true;
    }

    bool writeCompressedSparseMatSPHToBinary(const std::string& writePath, const SparseMatSPH& vecOfSparseMatSPH)
    {

        // First, serialize the nested sparse vectors into a continuous buffer
        const auto [totalBytes, fitsInSingle] = calculateTotalBytesSparseMatSPH(vecOfSparseMatSPH, false);

        bool success = false;

        if (fitsInSingle)
            success=  writeCompressedSparseMatSPHToBinarySingle(writePath, vecOfSparseMatSPH);
        else
            success = writeCompressedSparseMatSPHToBinaryBatches(writePath, vecOfSparseMatSPH);

        return success;
    }

    bool writeCompressedSparseMatSPHToBinaryBatches(const std::string& writePath, const SparseMatSPH& vecOfSparseMatSPH)
    {
        Log::info("writeCompressedSparseMatSPHToBinaryBatches: {}", writePath);

        // First, serialize the nested sparse vectors into a continuous buffer
        const auto [totalBytes, success] = calculateTotalBytesSparseMatSPH(vecOfSparseMatSPH, false);

        size_t numSparseMats = vecOfSparseMatSPH.size();
        size_t numChunks = static_cast<size_t>(std::ceil(static_cast<double>(totalBytes) / std::numeric_limits<int>::max()));
        numChunks = std::max(static_cast<size_t>(1), numChunks);

        // There is an assumption here that the non-zero entries are spread across rows uniformly
        // but that is not necessarily the case. Double the chunks to reduce probability of failure
        numChunks = numChunks * 2;

        __logMessage__("writeCompressedSparseMatSPHToBinaryBatches: dividing data in " + std::to_string(numChunks) + " chunks");

        // Write to file
        std::ofstream fout(writePath, std::ios::binary);
        if (!fout) {
            return false;
        }

        // Write the original total size (needed for decompression) and number of chunks
        fout.write(reinterpret_cast<const char*>(&totalBytes), sizeof(totalBytes));
        fout.write(reinterpret_cast<const char*>(&numSparseMats), sizeof(numSparseMats));
        fout.write(reinterpret_cast<const char*>(&numChunks), sizeof(numChunks));

        // see https://godbolt.org/z/hah9MsY63
        std::vector<size_t> counter(numSparseMats);
        std::iota(counter.begin(), counter.end(), 0);
        auto spansOfSparseMatSPH = divide_into_spans(counter, numChunks);

        assert(spansOfSparseMatSPH.size() == numChunks);

        for (size_t i = 0; i < spansOfSparseMatSPH.size(); ++i) {
            auto& currentCounter = spansOfSparseMatSPH[i];

            auto startIt = vecOfSparseMatSPH.begin() + currentCounter.front();
            auto endIt = vecOfSparseMatSPH.begin() + currentCounter.back() + 1;
            const auto [totalBytesSpan, success] = calculateTotalBytesSparseMatSPH(startIt, endIt);

            __debugMessage__("writeCompressedSparseMatSPHToBinaryBatches: chunk " + std::to_string(i) + " chunkOriginalSize " + std::to_string(totalBytesSpan));

            std::vector<char> serialBuffer(totalBytesSpan);
            char* writePtr = serialBuffer.data();

            // Write span size
            const size_t vecSize = currentCounter.size();
            memcpy(writePtr, &vecSize, sizeof(size_t));
            writePtr += sizeof(size_t);

            // Write sparse vectors
            for (auto sparseVecIt = startIt; sparseVecIt != endIt; sparseVecIt++) {
                // Write dimensions and non-zero count
                Eigen::Index rows = sparseVecIt->rows();
                Eigen::Index nnzs = sparseVecIt->nonZeros();
                memcpy(writePtr, &rows, sizeof(Eigen::Index));
                writePtr += sizeof(Eigen::Index);
                memcpy(writePtr, &nnzs, sizeof(Eigen::Index));
                writePtr += sizeof(Eigen::Index);

                // Write non-zero elements
                for (SparseVecSPH::InnerIterator it(*sparseVecIt); it; ++it) {
                    SparseVecSPH::StorageIndex index = it.index();
                    float value = it.value();
                    memcpy(writePtr, &index, sizeof(SparseVecSPH::StorageIndex));
                    writePtr += sizeof(SparseVecSPH::StorageIndex);
                    memcpy(writePtr, &value, sizeof(float));
                    writePtr += sizeof(float);
                }
            }

            // Calculate maximum compressed size
            const int maxDstSize = lz4_maxCompressedSize(static_cast<int>(totalBytesSpan));
            std::vector<char> compressedBuffer(maxDstSize);

            // Compress the serialized data
            int compressedSize = lz4_compress(
                serialBuffer.data(),
                compressedBuffer.data(),
                static_cast<int>(totalBytesSpan),
                maxDstSize
            );

            if (compressedSize <= 0) {
                return false; // Compression failed
            }

            // Write the compressed data
            uint32_t chunkCompressedSize = static_cast<uint32_t>(compressedSize);
            uint32_t chunkOriginalSize = static_cast<uint32_t>(totalBytesSpan);
            fout.write(reinterpret_cast<const char*>(&chunkCompressedSize), sizeof(chunkCompressedSize));
            fout.write(reinterpret_cast<const char*>(&chunkOriginalSize), sizeof(chunkOriginalSize));
            fout.write(compressedBuffer.data(), compressedSize);
        }

        fout.close();
        return true;
    }

    bool writeCompressedSparseMatSPHToBinarySingle(const std::string& writePath, const SparseMatSPH& vec)
    {
        Log::info("writeCompressedSparseMatSPHToBinarySingle: {}", writePath);

        // First, serialize the nested sparse vectors into a continuous buffer
        const auto [totalBytes, success] = calculateTotalBytesSparseMatSPH(vec);
        std::vector<char> serialBuffer(totalBytes);
        char* writePtr = serialBuffer.data();

        // Write Vec size
        const size_t vecSize = vec.size();
        memcpy(writePtr, &vecSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        // Write sparse vectors
        for (const auto& sparseVector : vec) {
            // Write dimensions and non-zero count
            Eigen::Index rows = sparseVector.rows();
            Eigen::Index nnzs = sparseVector.nonZeros();
            memcpy(writePtr, &rows, sizeof(Eigen::Index));
            writePtr += sizeof(Eigen::Index);
            memcpy(writePtr, &nnzs, sizeof(Eigen::Index));
            writePtr += sizeof(Eigen::Index);

            // Write non-zero elements
            for (SparseVecSPH::InnerIterator it(sparseVector); it; ++it) {
                SparseVecSPH::StorageIndex index = it.index();
                float value = it.value();
                memcpy(writePtr, &index, sizeof(SparseVecSPH::StorageIndex));
                writePtr += sizeof(SparseVecSPH::StorageIndex);
                memcpy(writePtr, &value, sizeof(float));
                writePtr += sizeof(float);
            }
        }

        // Calculate maximum compressed size
        const int maxDstSize = lz4_maxCompressedSize(static_cast<int>(totalBytes));
        std::vector<char> compressedBuffer(maxDstSize);

        // Compress the serialized data
        int compressedSize = lz4_compress(
            serialBuffer.data(),
            compressedBuffer.data(),
            static_cast<int>(totalBytes),
            maxDstSize
        );

        if (compressedSize <= 0) {
            return false; // Compression failed
        }

        // Write to file
        std::ofstream fout(writePath, std::ios::binary);
        if (!fout) {
            return false;
        }

        size_t numSparseMats = vec.size();
        size_t numChunks = 1;

        // Write the original total size (needed for decompression)
        fout.write(reinterpret_cast<const char*>(&totalBytes), sizeof(totalBytes));
        fout.write(reinterpret_cast<const char*>(&numSparseMats), sizeof(numSparseMats));
        fout.write(reinterpret_cast<const char*>(&numChunks), sizeof(numChunks));

        // Write the compressed data
        fout.write(reinterpret_cast<const char*>(&compressedSize), sizeof(compressedSize));
        fout.write(compressedBuffer.data(), compressedSize);

        fout.close();
        return true;
    }

    bool writeCompressedVecsOfSparseMatSPHToBinary(const std::string& writePath, const std::vector<SparseMatSPH>& vecOfSparseMatSPH)
    {
        for (size_t i = 0; i < vecOfSparseMatSPH.size(); i++)
        {
            Log::info("writeCompressedVecsOfSparseMatSPHToBinary: writing level {0}", i);
            bool success = writeCompressedSparseMatSPHToBinary(writePath + "_" + std::to_string(i), vecOfSparseMatSPH[i]);

            if (!success)
                return false;
        }

        return true;
    }

    bool writeCompressedVecOfSparseMatSPHToBinary(const std::string& writePath, const std::vector<SparseMatSPH>& vecOfSparseMatSPH)
    {
        // First, serialize the nested sparse vectors into a continuous buffer
        const size_t totalBytes = calculateTotalBytesVecOfSparseMatSPH(vecOfSparseMatSPH);
        std::vector<char> serialBuffer(totalBytes);
        char* writePtr = serialBuffer.data();

        // Write outermost size
        const size_t outermostSize = vecOfSparseMatSPH.size();
        memcpy(writePtr, &outermostSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        // Write middle vectors
        for (const auto& middleVector : vecOfSparseMatSPH) {
            // Write middle size
            const size_t middleSize = middleVector.size();
            memcpy(writePtr, &middleSize, sizeof(size_t));
            writePtr += sizeof(size_t);

            // Write sparse vectors
            for (const auto& sparseVector : middleVector) {
                // Write dimensions and non-zero count
                Eigen::Index rows = sparseVector.rows();
                Eigen::Index nnzs = sparseVector.nonZeros();
                memcpy(writePtr, &rows, sizeof(Eigen::Index));
                writePtr += sizeof(Eigen::Index);
                memcpy(writePtr, &nnzs, sizeof(Eigen::Index));
                writePtr += sizeof(Eigen::Index);

                // Write non-zero elements
                for (SparseVecSPH::InnerIterator it(sparseVector); it; ++it) {
                    SparseVecSPH::StorageIndex index = it.index();
                    float value = it.value();
                    memcpy(writePtr, &index, sizeof(SparseVecSPH::StorageIndex));
                    writePtr += sizeof(SparseVecSPH::StorageIndex);
                    memcpy(writePtr, &value, sizeof(float));
                    writePtr += sizeof(float);
                }
            }
        }

        // Calculate maximum compressed size
        const int maxDstSize = lz4_maxCompressedSize(static_cast<int>(totalBytes));
        std::vector<char> compressedBuffer(maxDstSize);

        // Compress the serialized data
        int compressedSize = lz4_compress(
            serialBuffer.data(),
            compressedBuffer.data(),
            static_cast<int>(totalBytes),
            maxDstSize
        );

        if (compressedSize <= 0) {
            return false; // Compression failed
        }

        // Write to file
        std::ofstream fout(writePath, std::ios::binary);
        if (!fout) {
            return false;
        }

        // Write the original total size (needed for decompression)
        fout.write(reinterpret_cast<const char*>(&totalBytes), sizeof(totalBytes));

        // Write the compressed data
        fout.write(compressedBuffer.data(), compressedSize);

        fout.close();
        return true;
    }

    bool writeGraphToBinary(const std::string& fileNameBase, const GraphView& g)
    {
        bool success = false;
        std::string fileName = "";

        auto checkWrite = [](bool s, const std::string& f) -> bool {
            if (!s) { Log::warn("Writing failed: file {0}", f); return false; }
            return true;
            };

        fileName = fileNameBase + "Distances.cache";
        Log::info("writeGraphToBinary: writing {0}", fileName);
        success = utils::writeVecToBinary(fileName, g.getKnnDistances());
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "Indices.cache";
        Log::info("writeGraphToBinary: writing {0}", fileName);
        success = utils::writeVecToBinary(fileName, g.getKnnIndices());
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "NNs.cache";
        Log::info("writeGraphToBinary: writing {0}", fileName);
        success = utils::writeVecToBinary(fileName, g.getNns());
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "Symmetric.cache";
        Log::info("writeGraphToBinary: writing {0}", fileName);
        std::vector<uint8_t> symmetric = { static_cast<uint8_t>(g.isSymmetric()) };
        success = utils::writeVecToBinary(fileName, symmetric);
        if (!checkWrite(success, fileName)) return success;

        return success;
    }

    bool writeCompressedGraphToBinary(const std::string& fileNameBase, const GraphView& g)
    {
        bool success = false;
        std::string fileName = "";

        auto checkWrite = [](bool s, const std::string& f) -> bool {
            if (!s) { Log::warn("Writing failed: file {0}", f); return false; }
            return true;
            };

        fileName = fileNameBase + "Distances.cache";
        Log::info("writeCompressedGraphToBinary: writing {0}", fileName);
        success = utils::writeCompressedVecToBinary(fileName, g.getKnnDistances());
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "Indices.cache";
        Log::info("writeCompressedGraphToBinary: writing {0}", fileName);
        success = utils::writeCompressedVecToBinary(fileName, g.getKnnIndices());
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "NNs.cache";
        Log::info("writeCompressedGraphToBinary: writing {0}", fileName);
        success = utils::writeVecToBinary(fileName, g.getNns());
        if (!checkWrite(success, fileName)) return success;

        fileName = fileNameBase + "Symmetric.cache";
        Log::info("writeCompressedGraphToBinary: writing {0}", fileName);
        std::vector<uint8_t> symmetric = { static_cast<uint8_t>(g.isSymmetric()) };
        success = utils::writeVecToBinary(fileName, symmetric);
        if (!checkWrite(success, fileName)) return success;

        return success;
    }

    bool writeCompressedSparseMatHDIToBinary(const std::string& writePath, const SparseMatHDI& vecOfSparseMatHDI)
    {
        // First, serialize the nested sparse vectors into a continuous buffer
        const auto [totalBytes, fitsInSingle] = calculateTotalBytesSparseMatHDI(vecOfSparseMatHDI, false);

        bool success = false;

        if (fitsInSingle)
            success = writeCompressedSparseMatHDIToBinarySingle(writePath, vecOfSparseMatHDI);
        else
            success = writeCompressedSparseMatHDIToBinaryBatches(writePath, vecOfSparseMatHDI);

        return success;
    }

    bool writeCompressedSparseMatHDIToBinarySingle(const std::string& writePath, const SparseMatHDI& vecOfSparseMatHDI)
    {
        // First, serialize the nested vectors of pairs into a continuous buffer
        const auto [totalBytes, fitsInSingle] = calculateTotalBytesSparseMatHDI(vecOfSparseMatHDI);
        std::vector<char> serialBuffer(totalBytes);
        char* writePtr = serialBuffer.data();

        assert(fitsInSingle);

        Log::info("writeCompressedSparseMatHDIToBinarySingle: writing...");

        // Write outer size
        const size_t outerSize = vecOfSparseMatHDI.size();
        memcpy(writePtr, &outerSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        // Write inner vectors
        for (const auto& innerVec : vecOfSparseMatHDI) {
            // Write inner size
            const size_t innerSize = innerVec.size();
            memcpy(writePtr, &innerSize, sizeof(size_t));
            writePtr += sizeof(size_t);

            // Write pairs
            for (const auto& pair : innerVec) {
                // Write first element
                memcpy(writePtr, &pair.first, sizeof(uint32_t));
                writePtr += sizeof(uint32_t);

                // Write second element
                memcpy(writePtr, &pair.second, sizeof(float));
                writePtr += sizeof(float);
            }
        }

        // Calculate maximum compressed size
        const int maxDstSize = lz4_maxCompressedSize(static_cast<int>(totalBytes));
        std::vector<char> compressedBuffer(maxDstSize);

        // Compress the serialized data
        int compressedSize = lz4_compress(
            serialBuffer.data(),
            compressedBuffer.data(),
            static_cast<int>(totalBytes),
            maxDstSize
        );

        if (compressedSize <= 0) {
            return false; // Compression failed
        }

        // Write to file
        std::ofstream fout(writePath, std::ios::binary);
        if (!fout) {
            return false;
        }

        // Write the original total size (needed for decompression)
        constexpr size_t numChunks = 1;
        fout.write(reinterpret_cast<const char*>(&totalBytes), sizeof(totalBytes));
        fout.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));
        fout.write(reinterpret_cast<const char*>(&numChunks), sizeof(numChunks));

        // Write the compressed data
        fout.write(reinterpret_cast<const char*>(&compressedSize), sizeof(compressedSize));
        fout.write(compressedBuffer.data(), compressedSize);

        fout.close();
        return true;
    }

    bool writeCompressedSparseMatHDIToBinaryBatches(const std::string& writePath, const SparseMatHDI& vecOfSparseMatHDI)
    {
        Log::info("writeCompressedSparseMatHDIToBinaryBatches: {}", writePath);

        // First, serialize the nested sparse vectors into a continuous buffer
        const auto [totalBytes, success] = calculateTotalBytesSparseMatHDI(vecOfSparseMatHDI, false);

        size_t numSparseMats = vecOfSparseMatHDI.size();
        size_t numChunks = static_cast<size_t>(std::ceil(static_cast<double>(totalBytes) / std::numeric_limits<int>::max()));
        numChunks = std::max(static_cast<size_t>(1), numChunks);

        // There is an assumption here that the non-zero entries are spread across rows uniformly
        // but that is not necessarily the case. Double the chunks to reduce probability of failure
        numChunks = numChunks * 2;

        Log::info("writeCompressedSparseMatHDIToBinaryBatches: compressing in {} chunks", numChunks);

        // Write to file
        std::ofstream fout(writePath, std::ios::binary);
        if (!fout) {
            return false;
        }

        // Write the original total size (needed for decompression) and number of chunks
        fout.write(reinterpret_cast<const char*>(&totalBytes), sizeof(totalBytes));
        fout.write(reinterpret_cast<const char*>(&numSparseMats), sizeof(numSparseMats));
        fout.write(reinterpret_cast<const char*>(&numChunks), sizeof(numChunks));

        // see https://godbolt.org/z/hah9MsY63
        std::vector<size_t> counter(numSparseMats);
        std::iota(counter.begin(), counter.end(), 0);
        auto spansOfSparseMatHDI = divide_into_spans(counter, numChunks);

        assert(spansOfSparseMatHDI.size() == numChunks);

        for (size_t i = 0; i < spansOfSparseMatHDI.size(); ++i) {
            auto& currentCounter = spansOfSparseMatHDI[i];

            auto startIt = vecOfSparseMatHDI.begin() + currentCounter.front();
            auto endIt = vecOfSparseMatHDI.begin() + currentCounter.back() + 1;
            const auto [totalBytesSpan, success] = calculateTotalBytesSparseMatHDI(startIt, endIt);

            std::vector<char> serialBuffer(totalBytesSpan);
            char* writePtr = serialBuffer.data();

            // Write span size
            const size_t vecSize = currentCounter.size();
            memcpy(writePtr, &vecSize, sizeof(size_t));
            writePtr += sizeof(size_t);

            // Write sparse vectors
            for (auto sparseVecIt = startIt; sparseVecIt != endIt; sparseVecIt++) {
                // Write inner size
                const size_t innerSize = sparseVecIt->size();
                memcpy(writePtr, &innerSize, sizeof(size_t));
                writePtr += sizeof(size_t);

                // Write pairs
                for (const auto& pair : sparseVecIt->memory()) {
                    // Write first element
                    memcpy(writePtr, &pair.first, sizeof(uint32_t));
                    writePtr += sizeof(uint32_t);

                    // Write second element
                    memcpy(writePtr, &pair.second, sizeof(float));
                    writePtr += sizeof(float);
                }
            }

            // Calculate maximum compressed size
            const int maxDstSize = lz4_maxCompressedSize(static_cast<int>(totalBytesSpan));
            std::vector<char> compressedBuffer(maxDstSize);

            // Compress the serialized data
            int compressedSize = lz4_compress(
                serialBuffer.data(),
                compressedBuffer.data(),
                static_cast<int>(totalBytesSpan),
                maxDstSize
            );

            if (compressedSize <= 0) {
                return false; // Compression failed
            }

            // Write the compressed data
            uint32_t chunkCompressedSize = static_cast<uint32_t>(compressedSize);
            uint32_t chunkOriginalSize = static_cast<uint32_t>(totalBytesSpan);
            fout.write(reinterpret_cast<const char*>(&chunkCompressedSize), sizeof(chunkCompressedSize));
            fout.write(reinterpret_cast<const char*>(&chunkOriginalSize), sizeof(chunkOriginalSize));
            fout.write(compressedBuffer.data(), compressedSize);
        }

        fout.close();
        return true;
    }

    /// /////// ///
    /// LOADING ///
    /// /////// ///

    bool loadVecOfSparseMatSPHFromBinary(const std::string& fileName, std::vector<SparseMatSPH>& vec)
    {
        std::ifstream fin(fileName.c_str(), std::ios::binary);
        if (!fin.is_open()) return false;

        // Read the size of the outermost vector
        std::size_t outermostSize = 0;
        fin.read(reinterpret_cast<char*>(&outermostSize), sizeof(std::size_t));

        // Read each middle vector
        vec.clear();
        vec.resize(outermostSize);
        for (auto& middleVector : vec) {
            // Read the size of the middle vector
            std::size_t middleSize = 0;
            fin.read(reinterpret_cast<char*>(&middleSize), sizeof(std::size_t));
            middleVector.resize(middleSize);

            // Read each inner (sparse) vector
            for (auto& sparseVector : middleVector) {
                // Resize the sparse vector and read its data
                Eigen::Index rows{}, nnzs{};
                fin.read(reinterpret_cast<char*>(&rows), sizeof(Eigen::Index));
                fin.read(reinterpret_cast<char*>(&nnzs), sizeof(Eigen::Index));

                sparseVector = Eigen::SparseVector<float>(rows);
                for (Eigen::Index i = 0; i < nnzs; ++i) {
                    SparseVecSPH::StorageIndex index{};
                    float value{};
                    fin.read(reinterpret_cast<char*>(&index), sizeof(SparseVecSPH::StorageIndex));
                    fin.read(reinterpret_cast<char*>(&value), sizeof(float));
                    sparseVector.coeffRef(index) = value;
                }

            }
        }

        fin.close();
        return true;

    }

    bool loadGraphFromBinary(const std::string& fileNameBase, Graph& g)
    {
        auto& distances = g.getKnnDistances();
        auto& indices = g.getKnnIndices();
        auto& nns = g.getNns();
        std::vector<uint8_t> symmetric;

        bool success = false;
        std::string fileName = "";

        auto checkLoad = [](bool s, const std::string& f) -> bool {
            if (!s) { Log::warn("Loading failed: file {0}", f); return false; }
            return true;
            };

        fileName = fileNameBase + "Distances.cache";
        Log::info("loadGraphFromBinary: loading {0}", fileName);
        success = utils::loadVecFromBinary(fileName, distances);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "Indices.cache";
        Log::info("loadGraphFromBinary: loading {0}", fileName);
        success = utils::loadVecFromBinary(fileName, indices);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "NNs.cache";
        Log::info("loadGraphFromBinary: loading {0}", fileName);
        success = utils::loadVecFromBinary(fileName, nns);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "Symmetric.cache";
        Log::info("loadGraphFromBinary: loading {0}", fileName);
        success = utils::loadVecFromBinary(fileName, symmetric);
        if (!checkLoad(success, fileName)) return success;

        assert(symmetric.size() == 1);

        g.symmetric = static_cast<bool>(symmetric.front());
        g.numPoints = nns.size();
        g.updateOffsets();

        assert(g.numPoints <= 0 || g.isValid());

        return success;
    }

    bool loadCompressedGraphFromBinary(const std::string& fileNameBase, Graph& g)
    {
        auto& distances = g.getKnnDistances();
        auto& indices = g.getKnnIndices();
        auto& nns = g.getNns();
        std::vector<uint8_t> symmetric;

        bool success = false;
        std::string fileName = "";

        auto checkLoad = [](bool s, const std::string& f) -> bool {
            if (!s) { Log::warn("Loading failed: file {0}", f); return false; }
            return true;
            };

        fileName = fileNameBase + "Distances.cache";
        Log::info("loadCompressedGraphFromBinary: loading {0}", fileName);
        success = utils::loadCompressedVecFromBinary(fileName, distances);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "Indices.cache";
        Log::info("loadCompressedGraphFromBinary: loading {0}", fileName);
        success = utils::loadCompressedVecFromBinary(fileName, indices);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "NNs.cache";
        Log::info("loadCompressedGraphFromBinary: loading {0}", fileName);
        success = utils::loadVecFromBinary(fileName, nns);
        if (!checkLoad(success, fileName)) return success;

        fileName = fileNameBase + "Symmetric.cache";
        Log::info("loadCompressedGraphFromBinary: loading {0}", fileName);
        success = utils::loadVecFromBinary(fileName, symmetric);
        if (!checkLoad(success, fileName)) return success;

        assert(symmetric.size() == 1);

        g.symmetric = static_cast<bool>(symmetric.front());
        g.numPoints = nns.size();
        g.updateOffsets();

        assert(g.numPoints <= 0 || g.isValid());

        return success;
    }

    bool loadCompressedSparseMatSPHFromBinary(const std::string& readPath, SparseMatSPH& vecOfSparseMatSPH)
    {
        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original total size
        size_t originalSize = 0;
        size_t numSparseMats = 0;
        size_t numChunks = 0;

        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));
        fin.read(reinterpret_cast<char*>(&numSparseMats), sizeof(numSparseMats));
        fin.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));
        fin.close();

        bool success = false;

        if (numChunks == 1)
            success = loadCompressedSparseMatSPHFromBinarySingle(readPath, vecOfSparseMatSPH);
        else
            success = loadCompressedSparseMatSPHFromBinaryBatches(readPath, vecOfSparseMatSPH);

        return success;
    }

    bool loadCompressedSparseMatSPHFromBinaryBatches(const std::string& readPath, SparseMatSPH& vecOfSparseMatSPH)
    {
        Log::info("loadCompressedSparseMatSPHFromBinaryBatches: {}", readPath);

        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original total size
        size_t originalSize = 0;
        size_t numSparseMats = 0;
        size_t numChunks = 0;

        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));
        fin.read(reinterpret_cast<char*>(&numSparseMats), sizeof(numSparseMats));
        fin.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));

        __logMessage__("loadCompressedSparseMatSPHFromBinaryBatches: load from " + std::to_string(numChunks) + " chunks");


        // Resize outermost vector
        vecOfSparseMatSPH.resize(numSparseMats);
        size_t rowCounter = 0;

        for (size_t i = 0; i < numChunks; ++i) {
            // Read the chunk meta data
            uint32_t chunkCompressedSize = 0;
            uint32_t chunkOriginalSize = 0;
            fin.read(reinterpret_cast<char*>(&chunkCompressedSize), sizeof(chunkCompressedSize));
            fin.read(reinterpret_cast<char*>(&chunkOriginalSize), sizeof(chunkOriginalSize));
            if (!fin) return false;

            __debugMessage__("loadCompressedSparseMatSPHFromBinaryBatches: chunk " + std::to_string(i) + " chunkOriginalSize " + std::to_string(chunkOriginalSize));

            // Read the compressed data
            std::vector<char> compressedBuffer(chunkCompressedSize);
            fin.read(compressedBuffer.data(), chunkCompressedSize);
            if (!fin) return false;

            // Prepare buffer for decompressed data
            std::vector<char> decompressedBuffer(chunkOriginalSize);

            // Decompress the data
            int decompressedSize = lz4_decompress(
                compressedBuffer.data(),
                decompressedBuffer.data(),
                static_cast<int>(chunkCompressedSize),
                static_cast<int>(chunkOriginalSize)
            );

            if (static_cast<size_t>(decompressedSize) != chunkOriginalSize) {
                return false;
            }

            // Read from decompressed buffer
            const char* readPtr = decompressedBuffer.data();

            // Read vec size
            size_t vecSize;
            memcpy(&vecSize, readPtr, sizeof(size_t));
            readPtr += sizeof(size_t);

            vecSize += rowCounter;

            // Read sparse vectors
            for (; rowCounter < vecSize; ++rowCounter) {
                // Read dimensions and non-zero count
                Eigen::Index rows, nnzs;
                memcpy(&rows, readPtr, sizeof(Eigen::Index));
                readPtr += sizeof(Eigen::Index);
                memcpy(&nnzs, readPtr, sizeof(Eigen::Index));
                readPtr += sizeof(Eigen::Index);

                // Create and fill sparse vector
                vecOfSparseMatSPH[rowCounter] = SparseVecSPH(rows);
                vecOfSparseMatSPH[rowCounter].reserve(nnzs);

                // Read non-zero elements
                for (Eigen::Index k = 0; k < nnzs; ++k) {
                    SparseVecSPH::StorageIndex index;
                    float value;
                    memcpy(&index, readPtr, sizeof(SparseVecSPH::StorageIndex));
                    readPtr += sizeof(SparseVecSPH::StorageIndex);
                    memcpy(&value, readPtr, sizeof(float));
                    readPtr += sizeof(float);

                    vecOfSparseMatSPH[rowCounter].insertBack(index) = value;
                }

                vecOfSparseMatSPH[rowCounter].finalize();
            }
        }

        fin.close();
        return true;
    }

    bool loadCompressedSparseMatSPHFromBinarySingle(const std::string& readPath, SparseMatSPH& vecOfSparseMatSPH)
    {
        Log::info("loadCompressedSparseMatSPHFromBinarySingle: {}", readPath);

        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original total size
        size_t originalSize = 0;
        size_t numSparseMats = 0;
        size_t numChunks = 0;

        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));
        fin.read(reinterpret_cast<char*>(&numSparseMats), sizeof(numSparseMats));
        fin.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));

        // Get the compressed size
        int compressedSize = 0;
        fin.read(reinterpret_cast<char*>(&compressedSize), sizeof(compressedSize));

        // Read the compressed data
        std::vector<char> compressedBuffer(compressedSize);
        fin.read(compressedBuffer.data(), compressedSize);

        // Prepare buffer for decompressed data
        std::vector<char> decompressedBuffer(originalSize);

        // Decompress the data
        int decompressedSize = lz4_decompress(
            compressedBuffer.data(),
            decompressedBuffer.data(),
            compressedSize,
            static_cast<int>(originalSize)
        );

        if (static_cast<size_t>(decompressedSize) != originalSize) {
            return false;
        }

        // Read from decompressed buffer
        const char* readPtr = decompressedBuffer.data();

        // Read vec size
        size_t vecSize;
        memcpy(&vecSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);

        // Resize outermost vector
        vecOfSparseMatSPH.resize(vecSize);

        // Read sparse vectors
        for (size_t i = 0; i < vecSize; ++i) {
            // Read dimensions and non-zero count
            Eigen::Index rows, nnzs;
            memcpy(&rows, readPtr, sizeof(Eigen::Index));
            readPtr += sizeof(Eigen::Index);
            memcpy(&nnzs, readPtr, sizeof(Eigen::Index));
            readPtr += sizeof(Eigen::Index);

            // Create and fill sparse vector
            vecOfSparseMatSPH[i] = SparseVecSPH(rows);
            vecOfSparseMatSPH[i].reserve(nnzs);

            // Read non-zero elements
            for (Eigen::Index k = 0; k < nnzs; ++k) {
                SparseVecSPH::StorageIndex index;
                float value;
                memcpy(&index, readPtr, sizeof(SparseVecSPH::StorageIndex));
                readPtr += sizeof(SparseVecSPH::StorageIndex);
                memcpy(&value, readPtr, sizeof(float));
                readPtr += sizeof(float);

                vecOfSparseMatSPH[i].insertBack(index) = value;
            }

            vecOfSparseMatSPH[i].finalize();
        }

        fin.close();
        return true;
    }

    bool loadCompressedVecsOfSparseMatSPHFromBinary(const std::string& readPath, std::vector<SparseMatSPH>& vecOfSparseMatSPH)
    {
        std::filesystem::path filePath(readPath);
        std::filesystem::path parentDir = filePath.parent_path();
        std::string targetFileName = filePath.stem().string();  // filename without extension

        // Iterate through each file in the parent directory
        size_t count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(parentDir)) {
            if (entry.is_regular_file()) {
                std::string entryFileName = entry.path().stem().string();  // get filename without extension

                // Check if it starts with the target filename
                if (entryFileName.find(targetFileName) == 0) {  // starts with targetFileName
                    ++count;
                }
            }
        }

        vecOfSparseMatSPH.clear();
        vecOfSparseMatSPH.resize(count);

        for (size_t i = 0; i < count; i++)
        {
            Log::info("loadCompressedVecsOfSparseMatSPHFromBinary: loading part {0}/{1}", i, count - 1);

            bool success = loadCompressedSparseMatSPHFromBinary(readPath + "_" + std::to_string(i), vecOfSparseMatSPH[i]);

            if (!success)
                return false;
        }

        return true;
    }

    bool loadCompressedVecOfSparseMatSPHFromBinary(const std::string& readPath, std::vector<SparseMatSPH>& vecOfSparseMatSPH)
    {
        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original total size
        size_t originalSize = 0;
        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));

        // Get the compressed size
        fin.seekg(0, std::ios::end);
        int compressedSize = static_cast<int>(fin.tellg()) - sizeof(originalSize);
        fin.seekg(sizeof(originalSize));

        // Read the compressed data
        std::vector<char> compressedBuffer(compressedSize);
        fin.read(compressedBuffer.data(), compressedSize);

        // Prepare buffer for decompressed data
        std::vector<char> decompressedBuffer(originalSize);

        // Decompress the data
        int decompressedSize = lz4_decompress(
            compressedBuffer.data(),
            decompressedBuffer.data(),
            compressedSize,
            static_cast<int>(originalSize)
        );

        if (static_cast<size_t>(decompressedSize) != originalSize) {
            return false;
        }

        // Read from decompressed buffer
        const char* readPtr = decompressedBuffer.data();

        // Read outermost size
        size_t outermostSize;
        memcpy(&outermostSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);

        // Resize outermost vector
        vecOfSparseMatSPH.resize(outermostSize);

        // Read middle vectors
        for (size_t i = 0; i < outermostSize; ++i) {
            // Read middle size
            size_t middleSize;
            memcpy(&middleSize, readPtr, sizeof(size_t));
            readPtr += sizeof(size_t);

            // Resize middle vector
            vecOfSparseMatSPH[i].resize(middleSize);

            // Read sparse vectors
            for (size_t j = 0; j < middleSize; ++j) {
                // Read dimensions and non-zero count
                Eigen::Index rows, nnzs;
                memcpy(&rows, readPtr, sizeof(Eigen::Index));
                readPtr += sizeof(Eigen::Index);
                memcpy(&nnzs, readPtr, sizeof(Eigen::Index));
                readPtr += sizeof(Eigen::Index);

                // Create and fill sparse vector
                vecOfSparseMatSPH[i][j] = SparseVecSPH(rows);
                vecOfSparseMatSPH[i][j].reserve(nnzs);

                // Read non-zero elements
                for (Eigen::Index k = 0; k < nnzs; ++k) {
                    SparseVecSPH::StorageIndex index;
                    float value;
                    memcpy(&index, readPtr, sizeof(SparseVecSPH::StorageIndex));
                    readPtr += sizeof(SparseVecSPH::StorageIndex);
                    memcpy(&value, readPtr, sizeof(float));
                    readPtr += sizeof(float);

                    vecOfSparseMatSPH[i][j].insertBack(index) = value;
                }

                vecOfSparseMatSPH[i][j].finalize();
            }
        }

        fin.close();
        return true;
    }

    bool loadCompressedSparseMatHDIFromBinary(const std::string& readPath, SparseMatHDI& vecOfSparseMatHDI) 
    {
        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original total size
        size_t originalSize = 0;
        size_t numSparseMats = 0;
        size_t numChunks = 0;

        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));
        fin.read(reinterpret_cast<char*>(&numSparseMats), sizeof(numSparseMats));
        fin.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));
        fin.close();

        bool success = false;

        if (numChunks == 1)
            success = loadCompressedSparseMatHDIFromBinarySingle(readPath, vecOfSparseMatHDI);
        else
            success = loadCompressedSparseMatHDIFromBinaryBatches(readPath, vecOfSparseMatHDI);

        return success;

    }

    bool loadCompressedSparseMatHDIFromBinaryBatches(const std::string& readPath, SparseMatHDI& vecOfSparseMatHDI)
    {
        Log::info("loadCompressedSparseMatHDIFromBinaryBatches: {}", readPath);

        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original total size
        size_t originalSize = 0;
        size_t numSparseMats = 0;
        size_t numChunks = 0;

        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));
        fin.read(reinterpret_cast<char*>(&numSparseMats), sizeof(numSparseMats));
        fin.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));

        assert(numChunks > 1);

        Log::info("loadCompressedSparseMatHDIFromBinaryBatches: decompressing from {} chunks", numChunks);

        // Resize outermost vector
        vecOfSparseMatHDI.resize(numSparseMats);
        size_t rowCounter = 0;

        for (size_t i = 0; i < numChunks; ++i) {
            // Read the chunk meta data
            uint32_t chunkCompressedSize = 0;
            uint32_t chunkOriginalSize = 0;
            fin.read(reinterpret_cast<char*>(&chunkCompressedSize), sizeof(chunkCompressedSize));
            fin.read(reinterpret_cast<char*>(&chunkOriginalSize), sizeof(chunkOriginalSize));
            if (!fin) return false;

            // Read the compressed data
            std::vector<char> compressedBuffer(chunkCompressedSize);
            fin.read(compressedBuffer.data(), chunkCompressedSize);
            if (!fin) return false;

            // Prepare buffer for decompressed data
            std::vector<char> decompressedBuffer(chunkOriginalSize);

            // Decompress the data
            int decompressedSize = lz4_decompress(
                compressedBuffer.data(),
                decompressedBuffer.data(),
                static_cast<int>(chunkCompressedSize),
                static_cast<int>(chunkOriginalSize)
            );

            if (static_cast<size_t>(decompressedSize) != chunkOriginalSize) {
                return false;
            }

            // Read from decompressed buffer
            const char* readPtr = decompressedBuffer.data();

            // Read vec size
            size_t vecSize;
            memcpy(&vecSize, readPtr, sizeof(size_t));
            readPtr += sizeof(size_t);

            vecSize += rowCounter;

            // Read sparse vectors
            for (; rowCounter < vecSize; ++rowCounter) {
                // Read inner size
                size_t innerSize;
                memcpy(&innerSize, readPtr, sizeof(size_t));
                readPtr += sizeof(size_t);

                // Resize inner vector
                auto& mem = vecOfSparseMatHDI[rowCounter].memory();
                mem.resize(innerSize);

                // Read pairs
                for (size_t j = 0; j < innerSize; ++j) {
                    // Read first element
                    uint32_t first = {};
                    memcpy(&first, readPtr, sizeof(uint32_t));
                    readPtr += sizeof(uint32_t);

                    // Read second element
                    float second = {};
                    memcpy(&second, readPtr, sizeof(float));
                    readPtr += sizeof(float);

                    // Store the pair
                    mem[j] = std::make_pair(first, second);
                }
            }

        }

        fin.close();
        return true;
    }

    bool loadCompressedSparseMatHDIFromBinarySingle(const std::string& readPath, SparseMatHDI& vecOfSparseMatHDI)
    {
        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original total size
        size_t originalSize = 0;
        size_t numSparseMats = 0;
        size_t numChunks = 0;

        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));
        fin.read(reinterpret_cast<char*>(&numSparseMats), sizeof(numSparseMats));
        fin.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));

        assert(numChunks == 1);

        Log::info("loadCompressedSparseMatHDIFromBinarySingle: reading...");

        // Read the compressed size
        int compressedSize = 0;
        fin.read(reinterpret_cast<char*>(&compressedSize), sizeof(compressedSize));

        // Read the compressed data
        std::vector<char> compressedBuffer(compressedSize);
        fin.read(compressedBuffer.data(), compressedSize);

        // Prepare buffer for decompressed data
        std::vector<char> decompressedBuffer(originalSize);

        // Decompress the data
        int decompressedSize = lz4_decompress(
            compressedBuffer.data(),
            decompressedBuffer.data(),
            compressedSize,
            static_cast<int>(originalSize)
        );

        if (decompressedSize != static_cast<int>(originalSize)) {
            return false;
        }

        // Read from decompressed buffer
        const char* readPtr = decompressedBuffer.data();

        // Read outer size
        size_t outerSize = 0;
        memcpy(&outerSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);

        // Resize outer vector
        vecOfSparseMatHDI.resize(outerSize);

        // Read inner vectors
        for (size_t i = 0; i < outerSize; ++i) {
            // Read inner size
            size_t innerSize;
            memcpy(&innerSize, readPtr, sizeof(size_t));
            readPtr += sizeof(size_t);

            // Resize inner vector
            auto& mem = vecOfSparseMatHDI[i].memory();
            mem.resize(innerSize);

            // Read pairs
            for (size_t j = 0; j < innerSize; ++j) {
                // Read first element
                uint32_t first = {};
                memcpy(&first, readPtr, sizeof(uint32_t));
                readPtr += sizeof(uint32_t);

                // Read second element
                float second = {};
                memcpy(&second, readPtr, sizeof(float));
                readPtr += sizeof(float);

                // Store the pair
                mem[j] = std::make_pair(first, second);
            }
        }

        fin.close();
        return true;
    }


} // namespace sph::utils
