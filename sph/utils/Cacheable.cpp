#include "Cacheable.hpp"

#include "Logger.hpp"

#include <string>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace sph::utils {


    Cacheable::Cacheable(const std::string& fileName)
    {
        _cachePath = std::filesystem::current_path() / _cacheSubfolder;
        _cacheFileName = fileName;
    }

    void Cacheable::setCachePathInfo(const std::string& path, const std::string& fileName, bool ignoreSubfolder, const std::string& customSubfolder)
    {
        _cachePath = std::filesystem::path(path);
        _cacheFileName = fileName;

        if (ignoreSubfolder)
            return;

        if (!customSubfolder.empty())
            _cachePath /= customSubfolder;
        else
            _cachePath /= _cacheSubfolder;

    }

    bool Cacheable::mayCache(const std::string& derivedClassName) const
    {
        std::string prefix = derivedClassName;
        if (prefix == "")
            prefix += "~";

        if (!_cacheIsActive)
        {
            Log::info(prefix + "::writeCache: Caching is not active. Use setCachingActive(true) if desired.");
            return false;
        }

        if (std::filesystem::exists(_cachePath))
            return true;

        if (std::filesystem::exists(_cachePath.parent_path()))
            if (std::filesystem::create_directory(_cachePath))
                return true;

        Log::info(prefix + "::writeCache: cannot create save path. No caching.");
        return false;
    }

    bool Cacheable::isVersionCompatible(const nlohmann::json& json) const
    {
        const auto versionField = "## VERSION ##";
        if (!json.contains(versionField))
        {
            Log::info("Cache does not contain a version. Require version {0}", _cacheParameterVersion);
            return false;
        }

        const std::string givenVersion = json[versionField];
        if (givenVersion != std::string(Cacheable::_cacheParameterVersion))
        {
            Log::info("Version of loaded cache file ({0}) differs from running library analysis version ({1}). Cannot load cache)", givenVersion, _cacheParameterVersion);
            return false;
        }

        return true;
    }

} // namespace sph::utils
