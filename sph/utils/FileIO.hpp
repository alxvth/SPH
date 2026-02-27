#pragma once

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <ios>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "CommonDefinitions.hpp"

namespace sph::utils {

    struct Graph;
    struct GraphView;

    void __logMessage__(const std::string& msg);
    void __debugMessage__(const std::string& msg);
    void __errorMessage__(const std::string& msg);
    void __errorMessageLoading__();

    // Helper function to calculate total bytes needed for a vector of vectors
    // Warns if totalBytes is larger than max(int), since lz4 handles only int
    template<typename T>
    size_t calculateTotalBytesOfVecOfVec(const std::vector<std::vector<T>>& vec) {
        size_t totalBytes = sizeof(size_t); // Outer size
        for (const auto& innerVec : vec) {
            totalBytes += sizeof(size_t); // Inner size
            totalBytes += innerVec.size() * sizeof(T); // Inner data
        }

        if (totalBytes > std::numeric_limits<int>::max())
            __errorMessage__("File to large for default lz4 compression. Proceeding, but expect errors and faulty files.");

        return totalBytes;
    }

    // Helper function to calculate total bytes needed for a vector of vector of vectors
    // Warns if totalBytes is larger than max(int), since lz4 handles only int
    template<typename T>
    size_t calculateTotalBytesOfVecOfVecOfVec(const std::vector<std::vector<std::vector<T>>>& vec) {
        size_t totalBytes = sizeof(size_t); // Outermost size
        for (const auto& middleVec : vec) {
            totalBytes += sizeof(size_t); // Middle size
            for (const auto& innerVec : middleVec) {
                totalBytes += sizeof(size_t); // Inner size
                totalBytes += innerVec.size() * sizeof(T); // Inner data
            }
        }

        if (totalBytes > std::numeric_limits<int>::max())
            __errorMessage__("File to large for default lz4 compression. Proceeding, but expect errors and faulty files.");

        return totalBytes;
    }

    std::vector<std::span<size_t>> divide_into_spans(std::vector<size_t>& data, size_t n);

    int lz4_maxCompressedSize(int totalBytes);

    int lz4_compress(const char* src, char* dst, int srcSize, int dstCapacity);

    int lz4_decompress(const char* src, char* dst, int compressedSize, int dstCapacity);


    /// /////// ///
    /// WRITING ///
    /// /////// ///

    //* Utility function to check whether a file opens correctly */
    template <typename StreamType,
    typename = std::enable_if<std::is_same<StreamType, std::ifstream>::value ||
                              std::is_same<StreamType, std::ofstream>::value>>
    inline bool fileOpens(StreamType& file)  {
        if (!file.is_open())
        {
            __errorMessageLoading__();
            return false;
        }

        return true;
    }

    template<typename T>
    bool writeVecToBinary(const std::string& writePath, const std::vector<T>& vec) {
        
        std::ofstream fout(writePath, std::ios::binary);
        if (!fileOpens(fout)) return false;

        // Write the contents of the vector to the file
        fout.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
        fout.close();

        return true;
    }

    template<typename T>
    bool writeVecOfVecToBinary(const std::string& writePath, const std::vector<std::vector<T>>& vec) {

        std::ofstream fout(writePath, std::ios::binary);
        if (!fileOpens(fout)) return false;

        // Write the size of the outer vector
        std::size_t outerSize = vec.size();
        fout.write(reinterpret_cast<const char*>(&outerSize), sizeof(std::size_t));

        // Write each inner vector's size and data
        for (const auto& innerVector : vec) {
            // Write the size of the inner vector
            std::size_t innerSize = innerVector.size();
            fout.write(reinterpret_cast<const char*>(&innerSize), sizeof(std::size_t));

            // Write the data of the inner vector
            fout.write(reinterpret_cast<const char*>(innerVector.data()), innerSize * sizeof(T));
        }

        fout.close();
        return true;
    }

    template<typename T>
    bool writeVecOfVecOfVecToBinary(const std::string& writePath, const std::vector<std::vector<std::vector<T>>>& vec) {

        std::ofstream fout(writePath, std::ios::binary);
        if (!fileOpens(fout)) return false;

        // Write the size of the outermost vector
        std::size_t outermostSize = vec.size();
        fout.write(reinterpret_cast<const char*>(&outermostSize), sizeof(std::size_t));

        // Write each middle vector
        for (const auto& middleVector : vec) {
            // Write the size of the middle vector
            std::size_t middleSize = middleVector.size();
            fout.write(reinterpret_cast<const char*>(&middleSize), sizeof(std::size_t));

            // Write each inner vector
            for (const auto& innerVector : middleVector) {
                // Write the size of the inner vector
                std::size_t innerSize = innerVector.size();
                fout.write(reinterpret_cast<const char*>(&innerSize), sizeof(std::size_t));

                // Write the data of the inner vector
                fout.write(reinterpret_cast<const char*>(innerVector.data()), innerSize * sizeof(T));
            }
        }

        fout.close();
        return true;
    }

    bool writeSparseMatHDIToBinary(const std::string& writePath, const SparseMatHDI& vec);

    bool writeSparseMatSPHToBinary(const std::string& writePath, const SparseMatSPH& vecOfSparseMatSPH);

    bool writeVecOfSparseMatSPHToBinary(const std::string& writePath, const std::vector<SparseMatSPH>& vecOfSparseMatSPH);

    bool writeGraphToBinary(const std::string& fileNameBase, const GraphView& g);

    template<typename T>
    bool writeCompressedVecToBinarySingle(const std::string& writePath, const std::vector<T>& vec) {
        // Calculate the maximum compressed size needed
        const size_t numItemsTotal  = vec.size();
        const size_t totalBytes     = numItemsTotal * sizeof(T);
        const int maxDstSize        = lz4_maxCompressedSize(static_cast<int>(totalBytes));
        const size_t numChunks      = 1;

        // Allocate buffer for compressed data
        std::vector<char> compressedBuffer(maxDstSize);

        // Compress the data
        int compressedSize = lz4_compress(
            reinterpret_cast<const char*>(vec.data()),
            compressedBuffer.data(),
            static_cast<int>(totalBytes),
            maxDstSize
        );

        if (compressedSize <= 0) {
            return false; // Compression failed
        }

        // Open file for writing
        std::ofstream fout(writePath, std::ios::binary);
        if (!fout) {
            return false;
        }

        // Write the original size
        fout.write(reinterpret_cast<const char*>(&totalBytes), sizeof(totalBytes));
        fout.write(reinterpret_cast<const char*>(&numItemsTotal), sizeof(numItemsTotal));
        fout.write(reinterpret_cast<const char*>(&numChunks), sizeof(numChunks));

        // Write the compressed data
        fout.write(reinterpret_cast<const char*>(&compressedSize), sizeof(compressedSize));
        fout.write(compressedBuffer.data(), compressedSize);

        fout.close();
        return true;
    }

    template<typename T>
    bool writeCompressedVecToBinaryBatches(const std::string& writePath, const std::vector<T>& vec) {

        const size_t numItemsTotal  = vec.size();
        const size_t totalBytes     = numItemsTotal * sizeof(T);

        size_t numChunksDiv         = static_cast<size_t>(std::ceil(static_cast<double>(totalBytes) / std::numeric_limits<int>::max()));
        size_t numChunks            = std::max(static_cast<size_t>(1), numChunksDiv);

        // There is an assumption here that the non-zero entries are spread across rows uniformly
        // but that is not necessarily the case. Double the chunks to reduce probability of failure
        numChunks = numChunks * 2;

        __logMessage__("writeCompressedVecToBinaryBatches: dividing data in " + std::to_string(numChunks) + " chunks");

        // Write to file
        std::ofstream fout(writePath, std::ios::binary);
        if (!fout) {
            return false;
        }

        // Write the original total size (needed for decompression) and number of chunks
        fout.write(reinterpret_cast<const char*>(&totalBytes), sizeof(totalBytes));
        fout.write(reinterpret_cast<const char*>(&numItemsTotal), sizeof(numItemsTotal));
        fout.write(reinterpret_cast<const char*>(&numChunks), sizeof(numChunks));

        // see https://godbolt.org/z/hah9MsY63
        std::vector<size_t> counter(vec.size());
        std::iota(counter.begin(), counter.end(), 0);
        auto spansOfVec = divide_into_spans(counter, numChunks);

        size_t accumSize = 0;

        assert(spansOfVec.size() == numChunks);

        for (size_t i = 0; i < numChunks; ++i) {
            auto& currentCounter = spansOfVec[i];
            
            const T* startIt = vec.data() + currentCounter.front();
            int chunkOriginalSize = static_cast<int>(currentCounter.size()) * sizeof(T);
            accumSize += currentCounter.size();

            __debugMessage__("writeCompressedVecToBinaryBatches: chunk " + std::to_string(i) + " chunkOriginalSize " + std::to_string(chunkOriginalSize));

            // Calculate maximum compressed size
            const int maxDstSize = lz4_maxCompressedSize(chunkOriginalSize);
            std::vector<char> compressedBuffer(maxDstSize);

            // Compress the serialized data
            int chunkCompressedSize = lz4_compress(
                reinterpret_cast<const char*>(startIt),
                compressedBuffer.data(),
                chunkOriginalSize,
                maxDstSize
            );

            if (chunkCompressedSize <= 0) {
                return false; // Compression failed
            }

            // Write the compressed data
            fout.write(reinterpret_cast<const char*>(&chunkCompressedSize), sizeof(chunkCompressedSize));
            fout.write(reinterpret_cast<const char*>(&chunkOriginalSize), sizeof(chunkOriginalSize));
            fout.write(compressedBuffer.data(), chunkCompressedSize);
        }

        assert(accumSize == numItemsTotal);

        fout.close();
        return accumSize == numItemsTotal;
    }

    template<typename T>
    bool writeCompressedVecToBinary(const std::string& writePath, const std::vector<T>& vec) {

        size_t totalBytes = vec.size() * sizeof(T);

        bool success = false;

        if (totalBytes < std::numeric_limits<int>::max())
            success = writeCompressedVecToBinarySingle(writePath, vec);
        else
            success = writeCompressedVecToBinaryBatches(writePath, vec);

        return success;
    }

    template<typename T>
    bool writeCompressedVecOfVecToBinary(const std::string& writePath, const std::vector<std::vector<T>>& vec) {
        // First, serialize the vector of vectors into a continuous buffer
        const size_t totalBytes = calculateTotalBytesOfVecOfVec(vec);
        std::vector<char> serialBuffer(totalBytes);
        char* writePtr = serialBuffer.data();

        // Write outer size
        const size_t outerSize = vec.size();
        memcpy(writePtr, &outerSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        // Write inner vectors
        for (const auto& innerVec : vec) {
            // Write inner size
            const size_t innerSize = innerVec.size();
            memcpy(writePtr, &innerSize, sizeof(size_t));
            writePtr += sizeof(size_t);

            // Write inner data
            memcpy(writePtr, innerVec.data(), innerSize * sizeof(T));
            writePtr += innerSize * sizeof(T);
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

    template<typename T>
    bool writeCompressedVecOfVecOfVecToBinary(const std::string& writePath,
        const std::vector<std::vector<std::vector<T>>>& vec) {
        // First, serialize the nested vectors into a continuous buffer
        const size_t totalBytes = calculateTotalBytesOfVecOfVecOfVec(vec);
        std::vector<char> serialBuffer(totalBytes);
        char* writePtr = serialBuffer.data();

        // Write outermost size
        const size_t outermostSize = vec.size();
        memcpy(writePtr, &outermostSize, sizeof(size_t));
        writePtr += sizeof(size_t);

        // Write middle vectors
        for (const auto& middleVec : vec) {
            // Write middle size
            const size_t middleSize = middleVec.size();
            memcpy(writePtr, &middleSize, sizeof(size_t));
            writePtr += sizeof(size_t);

            // Write inner vectors
            for (const auto& innerVec : middleVec) {
                // Write inner size
                const size_t innerSize = innerVec.size();
                memcpy(writePtr, &innerSize, sizeof(size_t));
                writePtr += sizeof(size_t);

                // Write inner data
                memcpy(writePtr, innerVec.data(), innerSize * sizeof(T));
                writePtr += innerSize * sizeof(T);
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

    bool writeCompressedSparseMatSPHToBinary(const std::string& writePath, const SparseMatSPH& vecOfSparseMatSPH);
    bool writeCompressedSparseMatSPHToBinaryBatches(const std::string& writePath, const SparseMatSPH& vecOfSparseMatSPH);
    bool writeCompressedSparseMatSPHToBinarySingle(const std::string& writePath, const SparseMatSPH& vecOfSparseMatSPH);

    // writes each SparseMatSPH into its own file
    bool writeCompressedVecsOfSparseMatSPHToBinary(const std::string& writePath, const std::vector<SparseMatSPH>& vecOfSparseMatSPH);

    // writes all SparseMatSPH into a single file
    bool writeCompressedVecOfSparseMatSPHToBinary(const std::string& writePath, const std::vector<SparseMatSPH>& vecOfSparseMatSPH);

    bool writeCompressedGraphToBinary(const std::string& fileNameBase, const GraphView& g);

    bool writeCompressedSparseMatHDIToBinary(const std::string& writePath, const SparseMatHDI& vecOfSparseMatHDI);
    bool writeCompressedSparseMatHDIToBinaryBatches(const std::string& writePath, const SparseMatHDI& vecOfSparseMatHDI);
    bool writeCompressedSparseMatHDIToBinarySingle(const std::string& writePath, const SparseMatHDI& vecOfSparseMatHDI);

    /// /////// ///
    /// LOADING ///
    /// /////// ///

    template<typename T>
    bool loadVecFromBinary(const std::string& fileName, std::vector<T>& vec)
    {
        std::ifstream fin(fileName.c_str(), std::ios::binary);
        if (!fin.is_open()) return false;

        // Determine the size of the file
        fin.seekg(0, std::ios::end);
        std::streamsize fileSize = fin.tellg();
        fin.seekg(0, std::ios::beg);

        // Resize the vector to fit the file data
        vec.clear();
        vec.resize(fileSize / sizeof(T));

        // Read the binary data into the vector
        fin.read(reinterpret_cast<char*>(vec.data()), fileSize);
        fin.close();

        return true;
    }

    template<typename T>
    bool loadVecOfVecFromBinary(const std::string& fileName, std::vector<std::vector<T>>& vec)
    {
        std::ifstream fin(fileName.c_str(), std::ios::binary);
        if (!fin.is_open()) return false;

        // Read the size of the outer vector
        std::size_t outerSize = 0;
        fin.read(reinterpret_cast<char*>(&outerSize), sizeof(std::size_t));

        // Read each inner vector's size and data
        vec.clear();
        vec.resize(outerSize);
        for (auto& innerVector : vec) {
            // Read the size of the inner vector
            std::size_t innerSize = 0;
            fin.read(reinterpret_cast<char*>(&innerSize), sizeof(std::size_t));

            // Resize the inner vector and read its data
            innerVector.resize(innerSize);
            fin.read(reinterpret_cast<char*>(innerVector.data()), innerSize * sizeof(T));
        }

        fin.close();
        return true;
    }

    template<typename T>
    bool loadVecOfVecOfVecFromBinary(const std::string& fileName, std::vector<std::vector<std::vector<T>>>& vec)
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

            // Read each inner vector
            middleVector.resize(middleSize);
            for (auto& innerVector : middleVector) {
                // Read the size of the inner vector
                std::size_t innerSize = 0;
                fin.read(reinterpret_cast<char*>(&innerSize), sizeof(std::size_t));

                // Resize the inner vector and read its data
                innerVector.resize(innerSize);
                fin.read(reinterpret_cast<char*>(innerVector.data()), innerSize * sizeof(T));
            }
        }

        fin.close();
        return true;
    }

    bool loadVecOfSparseMatSPHFromBinary(const std::string& fileName, std::vector<SparseMatSPH>& vec);

    bool loadGraphFromBinary(const std::string& fileNameBase, Graph& g);

    template<typename T>
    bool loadCompressedVecFromBinarySingle(const std::string& readPath, std::vector<T>& vec) {
        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original size
        size_t originalSize = 0;
        size_t numItemsTotal = 0;
        size_t numChunks = 0;

        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));
        fin.read(reinterpret_cast<char*>(&numItemsTotal), sizeof(numItemsTotal));
        fin.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));

        assert(numChunks == 1);
        assert(originalSize == numItemsTotal * sizeof(T));

        if (originalSize > std::numeric_limits<int>::max())
            __errorMessage__("loadCompressedVecOfVecFromBinary: data size might to to large to be loaded...");

        // Get the compressed size by checking remaining file size
        int compressedSize = 0;
        fin.read(reinterpret_cast<char*>(&compressedSize), sizeof(compressedSize));

        // Read the compressed data
        std::vector<char> compressedBuffer(compressedSize);
        fin.read(compressedBuffer.data(), compressedSize);

        // Resize the output vector
        vec.resize(numItemsTotal);

        // Decompress the data
        int decompressedSize = lz4_decompress(
            compressedBuffer.data(),
            reinterpret_cast<char*>(vec.data()),
            compressedSize,
            static_cast<int>(originalSize)
        );

        fin.close();
        return decompressedSize >= 0 && static_cast<size_t>(decompressedSize) == originalSize;
    }

    template<typename T>
    bool loadCompressedVecFromBinaryBatches(const std::string& readPath, std::vector<T>& vec) {
        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original size
        size_t originalSize = 0;
        size_t numItemsTotal = 0;
        size_t numChunks = 0;

        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));
        fin.read(reinterpret_cast<char*>(&numItemsTotal), sizeof(numItemsTotal));
        fin.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));

        assert(numChunks > 1);
        assert(originalSize == numItemsTotal * sizeof(T));

        __logMessage__("loadCompressedVecFromBinaryBatches: load from " + std::to_string(numChunks) + " chunks");

        vec.clear();
        vec.reserve(numItemsTotal);

        for (size_t i = 0; i < numChunks; ++i) {
            // Read the chunk meta data
            int chunkCompressedSize = 0;
            int chunkOriginalSize = 0;
            fin.read(reinterpret_cast<char*>(&chunkCompressedSize), sizeof(chunkCompressedSize));
            fin.read(reinterpret_cast<char*>(&chunkOriginalSize), sizeof(chunkOriginalSize));
            if (!fin) return false;

            __debugMessage__("loadCompressedVecFromBinaryBatches: chunk " + std::to_string(i) + " chunkOriginalSize " + std::to_string(chunkOriginalSize));

            // Read the compressed data
            std::vector<char> compressedBuffer(chunkCompressedSize, 0);
            fin.read(compressedBuffer.data(), chunkCompressedSize);
            if (!fin) return false;

            // Prepare buffer for decompressed data
            size_t chunkOriginalLength = chunkOriginalSize / sizeof(T);
            std::vector<T> decompressedBuffer(chunkOriginalLength);

            // Decompress the data
            int chunkDecompressedSize = lz4_decompress(
                compressedBuffer.data(),
                reinterpret_cast<char*>(decompressedBuffer.data()),
                chunkCompressedSize,
                chunkOriginalSize
            );

            if (chunkDecompressedSize != chunkOriginalSize) {
                return false;
            }
            
            // Move to actual data vector
            vec.insert(vec.end(),
                std::make_move_iterator(decompressedBuffer.begin()),
                std::make_move_iterator(decompressedBuffer.end()));
        }
        fin.close();

        return vec.size() == numItemsTotal;
    }

    template<typename T>
    bool loadCompressedVecFromBinary(const std::string& readPath, std::vector<T>& vec) {
        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original total size
        size_t originalSize = 0;
        size_t numItemsTotal = 0;
        size_t numChunks = 0;

        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));
        fin.read(reinterpret_cast<char*>(&numItemsTotal), sizeof(numItemsTotal));
        fin.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));
        fin.close();

        bool success = false;

        if (numChunks == 1)
            success = loadCompressedVecFromBinarySingle(readPath, vec);
        else
            success = loadCompressedVecFromBinaryBatches(readPath, vec);

        return success;
    }

    template<typename T>
    bool loadCompressedVecOfVecFromBinary(const std::string& readPath, std::vector<std::vector<T>>& vec) {
        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original total size
        size_t originalSize = 0;
        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));

        if (originalSize > std::numeric_limits<int>::max())
            __errorMessage__("loadCompressedVecOfVecFromBinary: data size might to to large to be loaded...");

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

        // Read outer size
        size_t outerSize = 0;
        memcpy(&outerSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);

        // Resize outer vector
        vec.resize(outerSize);

        // Read inner vectors
        for (size_t i = 0; i < outerSize; ++i) {
            // Read inner size
            size_t innerSize;
            memcpy(&innerSize, readPtr, sizeof(size_t));
            readPtr += sizeof(size_t);

            // Resize inner vector
            vec[i].resize(innerSize);

            // Read inner data
            memcpy(vec[i].data(), readPtr, innerSize * sizeof(T));
            readPtr += innerSize * sizeof(T);
        }

        fin.close();
        return true;
    }

    template<typename T>
    bool loadCompressedVecOfVecOfVecFromBinary(const std::string& readPath,
        std::vector<std::vector<std::vector<T>>>& vec) {
        std::ifstream fin(readPath, std::ios::binary);
        if (!fin) {
            return false;
        }

        // Read the original total size
        size_t originalSize = 0;
        fin.read(reinterpret_cast<char*>(&originalSize), sizeof(originalSize));

        if (originalSize > std::numeric_limits<int>::max())
            __errorMessage__("loadCompressedVecOfVecFromBinary: data size might to to large to be loaded...");

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
        size_t outermostSize = 0;
        memcpy(&outermostSize, readPtr, sizeof(size_t));
        readPtr += sizeof(size_t);

        // Resize outermost vector
        vec.resize(outermostSize);

        // Read middle vectors
        for (size_t i = 0; i < outermostSize; ++i) {
            // Read middle size
            size_t middleSize;
            memcpy(&middleSize, readPtr, sizeof(size_t));
            readPtr += sizeof(size_t);

            // Resize middle vector
            vec[i].resize(middleSize);

            // Read inner vectors
            for (size_t j = 0; j < middleSize; ++j) {
                // Read inner size
                size_t innerSize;
                memcpy(&innerSize, readPtr, sizeof(size_t));
                readPtr += sizeof(size_t);

                // Resize inner vector
                vec[i][j].resize(innerSize);

                // Read inner data
                memcpy(vec[i][j].data(), readPtr, innerSize * sizeof(T));
                readPtr += innerSize * sizeof(T);
            }
        }

        fin.close();
        return true;
    }

    bool loadCompressedSparseMatSPHFromBinary(const std::string& readPath, SparseMatSPH& vecOfSparseMatSPH);
    bool loadCompressedSparseMatSPHFromBinaryBatches(const std::string& readPath, SparseMatSPH& vecOfSparseMatSPH);
    bool loadCompressedSparseMatSPHFromBinarySingle(const std::string& readPath, SparseMatSPH& vecOfSparseMatSPH);

    bool loadCompressedVecsOfSparseMatSPHFromBinary(const std::string& readPath, std::vector<SparseMatSPH>& vecOfSparseMatSPH);

    bool loadCompressedVecOfSparseMatSPHFromBinary(const std::string& readPath, std::vector<SparseMatSPH>& vecOfSparseMatSPH);

    bool loadCompressedGraphFromBinary(const std::string& fileNameBase, Graph& g);

    bool loadCompressedSparseMatHDIFromBinary(const std::string& readPath, SparseMatHDI& vecOfSparseMatHDI);
    bool loadCompressedSparseMatHDIFromBinaryBatches(const std::string& readPath, SparseMatHDI& vecOfSparseMatHDI);
    bool loadCompressedSparseMatHDIFromBinarySingle(const std::string& readPath, SparseMatHDI& vecOfSparseMatHDI);

} // namespace sph::utils
