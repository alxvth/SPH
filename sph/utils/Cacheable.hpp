#pragma once

#include <filesystem>
#include <string>
#include <type_traits>

#include "Settings.hpp"

namespace sph::utils {

    class Cacheable
    {
    public:
        Cacheable(const std::string& fileName);

    public: // Setter
        void setCacheSettings(const CacheSettings& cs) {
            setCachePathInfo(cs.path, cs.fileName, cs.ignoreSubfolder, cs.customSubfolder);
            setCachingActive(cs.cacheActive);
        }

        void setCachePathInfo(const std::string& path, const std::string& fileName, bool ignoreSubfolder = false, const std::string& customSubfolder = "");
        void setCachingActive(bool active) { _cacheIsActive = active; }
        template <typename T>
            requires std::is_base_of_v<Cacheable, T>
        void setCachingDependency(const T* cacheable) { _cacheDependency = cacheable; }

    public: // Getter
        CacheSettings getCacheSettings() const {
            return { _cachePath.string() , _cacheFileName, _cacheIsActive };
        }

        std::string getSubfolder() const noexcept { return _cacheSubfolder; }
        bool getCachingActive() const noexcept { return _cacheIsActive; }
        bool getCacheIsValid() const noexcept { return _cacheIsValid; }
        const auto& getCachingFileName() const noexcept { return _cacheFileName; }
        const auto& getCachingPath() const noexcept { return _cachePath; }

    protected: // saving and loading
        virtual bool loadCache() = 0;
        virtual bool writeCache() const = 0;

        virtual bool checkCacheParameters(const std::string& fileName) const = 0;
        virtual bool writeCacheParameters(const std::string& fileName) const = 0;

        inline bool cacheDependencyIsValid() const noexcept { return (_cacheDependency == nullptr) ? true : _cacheDependency->getCacheIsValid(); }

        inline bool cachingFailure() noexcept { _cacheIsValid = false; return _cacheIsValid; };   /** sets _cacheIsValid false */
        inline bool cachingSuccess() noexcept { _cacheIsValid = true;  return _cacheIsValid; };   /** sets _cacheIsValid true */

        bool mayCache(const std::string& derivedClassName = "") const;

        bool isVersionCompatible(const nlohmann::json& json) const;

    protected:
        static constexpr auto   _cacheParameterVersion = "1.0";
        std::string             _cacheSubfolder = "sph-cache";
        std::filesystem::path   _cachePath = {};                    /** Path for saving and loading cache */
        std::string             _cacheFileName = {};                /** cachePath() + data name */
        bool                    _cacheIsActive = true;
        bool                    _cacheIsValid = false;
        const Cacheable*        _cacheDependency = nullptr;         /** if loading cache for previous computation failed, don't load cache for next */
    };

} // namespace sph::utils
