#include "EvalIO.hpp"

#include "Logger.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <ios>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <system_error>
#include <vector>

#ifdef _MSC_VER
#include <time.h>
#else
#include <ctime>
#include <cstring>
#endif

#include <nlohmann/json.hpp>

#include <jpeglib.h>
#include <png.h>
#include <tiffio.h>

namespace sph::utils
{

    namespace fs = std::filesystem;

    bool folderIsEmpty(const std::filesystem::path& p)
    {
        // Check if the directory exists
        if (!std::filesystem::exists(p))
            return true;

        // Check if the path is a directory
        if (!std::filesystem::is_directory(p))
            return true;

        // Iterate over the directory to check for files
        for (const auto& entry : std::filesystem::directory_iterator(p)) {
            if (std::filesystem::is_regular_file(entry.path())) {
                return false; // Found at least one file
            }
        }

        return true; // No files found in the directory
    }

    // Function to convert 16-bit float to 32-bit float
    float convert16To32BitFloat(uint16_t half) {
        // Extract the sign, exponent, and mantissa
        uint16_t sign = (half & 0x8000) >> 15;
        uint16_t exponent = (half & 0x7C00) >> 10;
        uint16_t mantissa = half & 0x03FF;

        // Handle special cases
        if (exponent == 0) {
            // Zero or subnormal
            if (mantissa == 0) {
                // Zero
                return sign ? -0.0f : 0.0f;
            }
            else {
                // Subnormal
                float result = static_cast<float>((mantissa / 1024.0f) * std::pow(2, -14));
                return sign ? -result : result;
            }
        }
        else if (exponent == 31) {
            // Infinity or NaN
            if (mantissa == 0) {
                // Infinity
                return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
            }
            else {
                // NaN
                return std::numeric_limits<float>::quiet_NaN();
            }
        }
        else {
            // Normalized number
            float result = static_cast<float>((1 + mantissa / 1024.0f) * std::pow(2, exponent - 15));
            return sign ? -result : result;
        }
    }


    Image loadTiffImage(const char* filename, bool verbose, bool flipUd) {
        Image image = { 0, 0, {} };

        TIFF* tiff = TIFFOpen(filename, "r");
        if (!tiff) {
            std::cerr << "Could not open input file " << filename << std::endl;
            return image;
        }

        uint32_t width, height;
        TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);

        image.width = width;
        image.height = height;
        image.data.resize(static_cast<uint64_t>(width) * height);

        // Ensure the TIFF file is in the format we expect
        uint16_t samplesPerPixel(UINT16_MAX), bitsPerSample(UINT16_MAX), sampleFormat(UINT16_MAX);
        TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
        TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
        TIFFGetField(tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat);

        if ((bitsPerSample != 32 && bitsPerSample != 16 && bitsPerSample != 8) || (sampleFormat == SAMPLEFORMAT_IEEEFP && bitsPerSample != 32)) {
            std::cerr << "Unsupported TIFF format" << std::endl;
            TIFFClose(tiff);
            return image;
        }

        if (verbose)
        {
            if (samplesPerPixel == UINT16_MAX)
                std::cout << "loadTiffImage: tag samplesPerPixel not found" << std::endl;
            if (sampleFormat == UINT16_MAX)
                std::cout << "loadTiffImage: tag sampleFormat not found" << std::endl;
        }

        auto getMultiplier = [flipUd, height](uint32_t row) -> size_t {
            if (flipUd)
                return static_cast<size_t>(height) - 1 - row;
            else
                return static_cast<size_t>(row);
            };

        // Read the data
        if (sampleFormat == SAMPLEFORMAT_IEEEFP)
        {
            assert(bitsPerSample == 32);

            for (uint32_t row = 0; row < height; ++row)
                TIFFReadScanline(tiff, &image.data[getMultiplier(row) * width], row);

            // assume 32 bit, in future maybe use convert16To32BitFloat()
            
        }
        else // assume uint even if sampleFormat was not set
        {
            if (bitsPerSample == 32)
            {
                std::vector<uint32_t> buffer(width);
                for (uint32_t row = 0; row < height; ++row)
                {
                    TIFFReadScanline(tiff, buffer.data(), row);
                    for (uint32_t col = 0; col < width; ++col) {
                        image.data[getMultiplier(row) * width + col] = static_cast<float>(buffer[col]);
                    }
                }
            }
            else if (bitsPerSample == 16)
            {
                std::vector<uint16_t> buffer(width);
                for (uint32_t row = 0; row < height; ++row)
                {
                    TIFFReadScanline(tiff, buffer.data(), row);
                    for (uint32_t col = 0; col < width; ++col) {
                        image.data[getMultiplier(row) * width + col] = static_cast<float>(buffer[col]);
                    }
                }
            }
            else // bitsPerSample == 8
            {
                std::vector<uint8_t> buffer(width);
                for (uint32_t row = 0; row < height; ++row)
                {
                    TIFFReadScanline(tiff, buffer.data(), row);
                    for (uint32_t col = 0; col < width; ++col) {
                        image.data[getMultiplier(row) * width + col] = static_cast<float>(buffer[col]);
                    }
                }
            }

        }

        TIFFClose(tiff);
        return image;
    }

    ImageStack loadTiffImageStack(const std::filesystem::path& directory)
    {
        assert(!directory.has_extension());

        const auto tiffFiles = findTiffFiles(directory);

        ImageStack imageStack = { 0, 0, {} };

        for (const auto& tiffFile : tiffFiles)
        {
            auto tiff = loadTiffImage(tiffFile.string().c_str());

            if (imageStack.height == 0 && imageStack.width == 0)
            {
                imageStack.height = tiff.height;
                imageStack.width = tiff.width;
            }
            else
            {
                if (imageStack.height != tiff.height || imageStack.width != tiff.width)
                {
                    std::cerr << "loadTiffImageStack:: inconsistent dimensions of images in " << directory.string() << std::endl;
                    break;
                }
            }

            imageStack.data.dataVec.insert(imageStack.data.dataVec.end(),
                std::make_move_iterator(tiff.data.begin()),
                std::make_move_iterator(tiff.data.end()));
        }

        imageStack.data.numDimensions = static_cast<uint32_t>(tiffFiles.size());
        imageStack.data.numPoints = static_cast<uint64_t>(imageStack.height) * imageStack.width;

        // images are loaded as: all dim 0 values, all dim 1 values etc.
        // but we want p0_d0, p0_d1, p0_d2, ..., p1_d0, p1_d1, p1_d2, ...
        reorderImageDataVector(imageStack);

        return imageStack;
    }

    void reorderImageDataVector(ImageStack& stack) {
        
        std::vector<float> reorderedVec(stack.data.dataVec.size());

        for (int64_t i = 0; i < stack.data.numPoints; ++i) {
            for (int64_t j = 0; j < stack.data.numDimensions; ++j) {
                reorderedVec[stack.data.numDimensions * i + j] = stack.data.dataVec[stack.data.numPoints * j + i];
            }
        }

        stack.data.dataVec = std::move(reorderedVec);
    }

    void saveVectorAsTiff(const Image& img, const char* filename, bool flipUd) {
        // Open the TIFF file for writing
        TIFF* tiff = TIFFOpen(filename, "w");
        if (!tiff) {
            std::cerr << "Could not open output file " << filename << std::endl;
            return;
        }

        // Set TIFF fields
        TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, img.width);
        TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, img.height);
        TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 32);
        TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP); // Floating point samples

        std::vector<float> buffer(img.width);

        auto getMultiplier = [flipUd, img](uint32_t row) -> size_t {
            if (flipUd)
                return static_cast<int64_t>(img.height) - 1 - row;
            else
                return static_cast<int64_t>(row);
            };

        // Write the data to the TIFF file
        for (uint32_t row = 0; row < img.height; ++row) {

            for (size_t i = 0; i < img.width; ++i)
                buffer[i] = img.data[getMultiplier(row) * img.width + i];

            TIFFWriteScanline(tiff, (void*)buffer.data(), row);
        }

        // Close the TIFF file
        TIFFClose(tiff);
    }

    std::vector<fs::path> findTiffFiles(const fs::path& directory) {
        std::vector<fs::path> tiff_files;
        try {
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.is_regular_file() && (entry.path().extension() == ".tiff" || entry.path().extension() == ".tif")) {
                    tiff_files.push_back(entry.path());
                }
            }
        }
        catch (const fs::filesystem_error& e) {
            std::cerr << "Filesystem error: " << e.what() << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "General error: " << e.what() << std::endl;
        }
        return tiff_files;
    }

    bool ensurePathExists(const std::filesystem::path& p) {

        bool success = false;

        if (std::filesystem::exists(p))
            success = true;
        else
            success = std::filesystem::create_directories(p);

        return success;
    }


    static bool loadJPEG(const char* filename, std::vector<uint8_t>& image_data, int& width, int& height) {
        FILE* infile = std::fopen(filename, "rb");
        if (!infile) {
            Log::error("loadJPEG: Could not open file {}", filename);
            return false;
        }

        // Decompression struct and error handling.
        struct jpeg_decompress_struct cinfo = {};
        struct jpeg_error_mgr jerr = {};

        cinfo.err = jpeg_std_error(&jerr);  // Set up error handling
        jpeg_create_decompress(&cinfo);     // Initialize the decompression object
        jpeg_stdio_src(&cinfo, infile);     // Specify data source (file)

        // Read the header of the JPEG file.
        jpeg_read_header(&cinfo, true);
        jpeg_start_decompress(&cinfo);      // Start the decompression process

        // Get image dimensions
        width = cinfo.output_width;
        height = cinfo.output_height;
        int channels = cinfo.output_components;

        if (channels != 3)
            Log::warn("loadRGBImage: Not 3 channels but {}. Things might fail.", channels);

        // Allocate buffer for the decompressed image.
        size_t row_stride = static_cast<size_t>(width) * channels;
        image_data.resize(row_stride * height);

        // Read the scanlines one at a time into the buffer.
        while (cinfo.output_scanline < static_cast<JDIMENSION>(height)) {
            unsigned char* buffer_array[1]{};
            buffer_array[0] = &image_data[cinfo.output_scanline * row_stride];
            jpeg_read_scanlines(&cinfo, buffer_array, 1);
        }

        // Finish decompression and cleanup.
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        std::fclose(infile);

        return true;
    }

    static bool loadPNG(const char* filename, std::vector<uint8_t>& image_data, int& width, int& height) {
        FILE* fp = std::fopen(filename, "rb");
        if (!fp) {
            std::cerr << "Error: Couldn't open file " << filename << std::endl;
            return false;
        }

        // Initialize libpng read structs.
        png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (!png) {
            std::fclose(fp);
            Log::error("loadJPEG: Couldn't create PNG read struct");
            return false;
        }

        png_infop info = png_create_info_struct(png);
        if (!info) {
            png_destroy_read_struct(&png, NULL, NULL);
            std::fclose(fp);
            Log::error("loadJPEG: Couldn't create PNG info struct");
            return false;
        }

        if (setjmp(png_jmpbuf(png))) {
            png_destroy_read_struct(&png, &info, NULL);
            std::fclose(fp);
            Log::error("loadJPEG: Error during setup");
            return false;
        }

        // Initialize file IO
        png_init_io(png, fp);

        // Read the PNG header
        png_read_info(png, info);

        // Get image details
        width = png_get_image_width(png, info);
        height = png_get_image_height(png, info);
        int channels = png_get_channels(png, info);
        png_byte color_type = png_get_color_type(png, info);
        png_byte bit_depth = png_get_bit_depth(png, info);

        // Handle different bit depths (for simplicity, assume 8-bit depth here).
        if (bit_depth == 16)
            png_set_strip_16(png);

        // Convert palette or grayscale images to RGB or RGBA.
        if (color_type == PNG_COLOR_TYPE_PALETTE)
            png_set_palette_to_rgb(png);
        if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
            png_set_expand_gray_1_2_4_to_8(png);

        if (png_get_valid(png, info, PNG_INFO_tRNS) || channels > 3)
            Log::warn("loadPNG: we don't handle transparency here. More than three channels. Things might go wrong.");

        // Update info structure after applying transformations.
        png_read_update_info(png, info);

        // Read image into memory.
        size_t row_bytes = png_get_rowbytes(png, info);
        image_data.resize(row_bytes * height);
        std::vector<png_bytep> row_pointers(height);

        for (int y = 0; y < height; ++y) {
            row_pointers[y] = &image_data[y * row_bytes];
        }

        png_read_image(png, row_pointers.data());

        // Cleanup
        png_destroy_read_struct(&png, &info, NULL);
        std::fclose(fp);

        return true;
    }


    ImageRGB loadRGBImage(const std::filesystem::path& p)
    {
        Log::info("loadRGBImage: Loading file {}", p.string().c_str());

        int width{}, height{};
        std::vector<uint8_t> image_data;
        bool readSuccess = false;

        const auto fileExtension = p.extension();
        if(fileExtension == ".jpeg" || fileExtension == ".jpg")
            readSuccess = loadJPEG(p.string().c_str(), image_data, width, height);
        else if(fileExtension == ".png")
            readSuccess = loadPNG(p.string().c_str(), image_data, width, height);
        else
        {
            Log::error("loadRGBImage: Unknown file extension {}. Not loading file.", fileExtension.string());
            return {};
        }

        if(!readSuccess)
        {
            Log::error("loadRGBImage: Failed to load image.");
            return {};
        }

        return { static_cast<uint32_t>(width), static_cast<uint32_t>(height), image_data };
    }

    Image convertRGBtoFloatImage(const ImageRGB& img)
    {
        Image outImg;

        outImg.data.reserve(img.data.size());
        std::transform(img.data.begin(), img.data.end(), std::back_inserter(outImg.data),
            [](uint8_t value) { return static_cast<float>(value); });

        outImg.height = img.height;
        outImg.width = img.width;

        return outImg;
    }

    ImageStack loadRGBdata(const std::filesystem::path& p) 
    {
        assert(p.has_extension());

        ImageStack imgData;

        auto imgRGB = loadRGBImage(p);
        auto img = convertRGBtoFloatImage(imgRGB);

        imgData.height = img.height;
        imgData.width = img.width;

        imgData.data.numDimensions = 3;
        imgData.data.numPoints = static_cast<uint64_t>(imgData.width) * imgData.height;
        imgData.data.dataVec = std::move(img.data);

        return imgData;
    }

    void saveLevelImages(size_t numLevels, const utils::Hierarchy& h, const ImageStack& imgStack, const std::filesystem::path& cachePath, bool flipUd) {
        std::vector<float> componentIDs;

        // randomly shuffle component IDs for more distinct color mapping of spatial neighbors
        std::mt19937 g;
        g.seed(1);

        for (int64_t level = 0; level < static_cast<int64_t>(numLevels); level++) {

            std::vector<float> shuffledIDs(h.numComponentsOn(level));
            std::iota(shuffledIDs.begin(), shuffledIDs.end(), 0.f);
            std::shuffle(shuffledIDs.begin(), shuffledIDs.end(), g);

            componentIDs.clear();
            componentIDs.resize(imgStack.data.numPoints, 0);

            for (int64_t point = 0; point < imgStack.data.numPoints; point++)
                componentIDs[point] = shuffledIDs[h.pixelComponentsOn(level)[point]];

            Image levelIdImg;
            levelIdImg.width = imgStack.width;
            levelIdImg.height = imgStack.height;
            std::swap(levelIdImg.data, componentIDs);
            const auto saveImgName = cachePath / ("ids_" + std::to_string(level) + ".tiff");
            Log::info("saveLevelImages: write to {}", saveImgName.string().c_str());
            saveVectorAsTiff(levelIdImg, saveImgName.string().c_str(), flipUd);
        }
    }

    void saveSingleImage(const std::vector<int64_t>& values, uint32_t width, uint32_t height, const std::filesystem::path& path, bool flipUd, bool shuffle) {

        Image levelIdImg;
        levelIdImg.width = width;
        levelIdImg.height = height;
        levelIdImg.data.resize(values.size());
        std::transform(values.begin(), values.end(), levelIdImg.data.begin(),
            [](int64_t i) {
                return static_cast<float>(i);  // Explicit cast to float
            });

        if(shuffle)
        {
            // randomly shuffle values IDs for more distinct color mapping of spatial neighbors
            std::mt19937 g;
            g.seed(1);

            std::shuffle(levelIdImg.data.begin(), levelIdImg.data.end(), g);
        }

        Log::info("saveSingleImage: write to {}", path.string().c_str());
        saveVectorAsTiff(levelIdImg, path.string().c_str(), flipUd);
    }

    bool saveCurrentSettings(const std::filesystem::path& fileName, const NearestNeighborsSettings& nns, const ImageHierarchySettings& ihs, const utils::RandomWalkSettings& rws, const LevelSimilaritiesSettings& lss)
    {
        // store parameters in json file
        nlohmann::json parameters;

        addToJson(nns, parameters);
        addToJson(ihs, parameters);
        addToJson(rws, parameters);
        addToJson(lss, parameters);

        // Write to file
        return writeJsonToDisk(fileName.string(), parameters);
    }

    bool saveSettingHashes(const std::filesystem::path& fileName, const std::vector<std::pair<std::string, std::string>>& settingHashes)
    {
        nlohmann::json folderNames;

        for (const auto& [ID, settingsHash] : settingHashes)
            folderNames[ID] = settingsHash;

        return writeJsonToDisk(fileName.string(), folderNames);
    }

    std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
        size_t start_pos = 0;
        while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
            str.replace(start_pos, from.length(), to);
            start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
        }
        return str;
    }

    std::string getCurrentDateTimeHuman() {
        const auto now  = std::chrono::system_clock::now();
        const auto time = std::chrono::system_clock::to_time_t(now);
        char timeBuff[30];

#ifdef _MSC_VER
        ctime_s(timeBuff, sizeof(timeBuff), &time);
#else
        std::strncpy(timeBuff, std::ctime(&time), sizeof(timeBuff) - 1);
        timeBuff[sizeof(timeBuff) - 1] = '\0'; // Ensure null-termination
#endif
        return std::string(timeBuff);
    }

    std::string getCurrentDateTimeNumbers() {
        // Get the current time as a time_point
        const auto now = std::chrono::system_clock::now();

        // Convert the time_point to time_t, which represents the time as a number of seconds since epoch
        const auto time = std::chrono::system_clock::to_time_t(now);

        // Convert to local time (struct tm)
        std::tm localTime;
#ifdef _MSC_VER
        localtime_s(&localTime, &time); // For MSVC (Windows)
#else
        localTime = *std::localtime(&time); // For Linux/Unix
#endif

        // Create a stringstream to hold the formatted date and time
        std::stringstream ss;

        // Use put_time to format the date and time as "year-month-day-hour-minute-second"
        ss << std::put_time(&localTime, "%Y-%m-%d-%H-%M-%S");

        // Return the formatted string
        return ss.str();
    }

    // Ensure same hash on all compilers, use Fowler-Noll-Vo hash function (same as MSVC STL)
    static std::size_t fnv1a_hash(const std::string& str) noexcept {
        constexpr size_t FNV_offset_basis    = 14695981039346656037ULL;
        constexpr size_t FNV_prime           = 1099511628211ULL;

        std::size_t hash = FNV_offset_basis;

        for (char c : str) {
            hash ^= static_cast<std::size_t>(c);
            hash *= FNV_prime;
        }

        return hash;
    }

    std::string createHash(const std::string& input)
    {
        // Convert the hash value to a hexadecimal string
        std::stringstream ss;
        ss << std::hex << fnv1a_hash(input);

        return ss.str();
    }

    bool copyFile(const std::filesystem::path& in, const std::filesystem::path& out)
    {
        std::error_code ec;
        fs::copy_file(in, out, fs::copy_options::overwrite_existing, ec);

        return ec.value() != 0;
    }

} // sph::utils
