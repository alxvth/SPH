#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "Data.hpp"
#include "Hierarchy.hpp"
#include "Settings.hpp"

namespace sph::utils
{
    /// ////////// ///
    /// Filesystem ///
    /// ////////// ///

    bool ensurePathExists(const std::filesystem::path& p);

    bool folderIsEmpty(const std::filesystem::path& p);

    bool copyFile(const std::filesystem::path& in, const std::filesystem::path& out);

    /// ////////// ///
    ///   Images   ///
    /// ////////// ///

    struct Image {
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<float> data = {};
    };

    struct ImageRGB {
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<uint8_t> data = {};
    };

    struct ImageStack {
        uint32_t width = 0;
        uint32_t height = 0;
        utils::Data data = {};
    };

    // data starts in lower left corner of image
    // currently handles single channel images, float32, uint16 and uint32
    Image loadTiffImage(const char* filename, bool verbose = false, bool flipUd = false);

    // directory should contain single-channel tiff files of the same shape
    // data will be [image0, image1, ... imageN]
    ImageStack loadTiffImageStack(const std::filesystem::path& directory);

    // saves as float32
    void saveVectorAsTiff(const Image& img, const char* filename, bool flipUd = false);

    // wraps saveVectorAsTiff
    void saveSingleImage(const std::vector<int64_t>& values, uint32_t width, uint32_t height, const std::filesystem::path& path, bool flipUd = false, bool shuffle = true);

    // wraps saveVectorAsTiff
    void saveLevelImages(size_t numLevels, const utils::Hierarchy& h, const ImageStack& imgStack, const std::filesystem::path& cachePath, bool flipUd = false);

    // reorders the above [image0, image1, ... imageN] to [attributes1, attributes2, ..., attributesN]
    void reorderImageDataVector(ImageStack& stack);

    std::vector<std::filesystem::path> findTiffFiles(const std::filesystem::path& directory);
    
    // Can open jpeg and png files
    ImageRGB loadRGBImage(const std::filesystem::path& p);

    Image convertRGBtoFloatImage(const ImageRGB& img);

    // opens jpeg and png files, converts them form uint8_t to float
    ImageStack loadRGBdata(const std::filesystem::path& p);


    /// //////////// ///
    ///   Settings   ///
    /// //////////// ///

    bool saveCurrentSettings(const std::filesystem::path& fileName, const NearestNeighborsSettings& nns, const ImageHierarchySettings& ihs, const utils::RandomWalkSettings& rws, const LevelSimilaritiesSettings& lss);

    bool saveSettingHashes(const std::filesystem::path& fileName, const std::vector<std::pair<std::string, std::string>>& settingHashes);

    /// ////////// ///
    ///    Misc    ///
    /// ////////// ///

    // returns e.g. "Wed Oct  2 11:58:16 2024"
    std::string getCurrentDateTimeHuman();

    // returns e.g. "2024-10-02-11-58"
    std::string getCurrentDateTimeNumbers();

    std::string createHash(const std::string& input);

    inline std::string createShortHash(const std::string& input, size_t length = 8)
    {
        std::string hashString = createHash(input);
        return hashString.substr(0, std::min(length, hashString.size()));
    }

    // Function to convert 16-bit float to 32-bit float
    float convert16To32BitFloat(uint16_t half);

    std::string replaceAll(std::string str, const std::string& from, const std::string& to);

} // namespace sph::utils
