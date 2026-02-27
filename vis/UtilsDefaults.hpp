#pragma once

#include <sph/utils/TestData.hpp>

#include <implot.h>
#include <tinycolormap.hpp>

#include <utility>
#include <vector>

namespace vis
{
	constexpr float NoiseDefaultSwissRole = 0.4f;
	constexpr float NoiseDefaultSCurve = 0.1f;
	constexpr float NoiseDefaultGaussians = 0.1f;

	inline float testDataNoise(const sph::utils::TestData& dataType)
	{
		float noise = 0.f;

		switch (dataType) {
		case sph::utils::TestData::SwissRole:
			noise = NoiseDefaultSwissRole;
			break;
		case sph::utils::TestData::SCurve:
			noise = NoiseDefaultSCurve;
			break;
		case sph::utils::TestData::Gaussians3D:
			noise = NoiseDefaultGaussians;
			break;
		}

		return noise;
	}

	static std::vector<float> gaussian3DCenters =
	{   -5.f, -5.f, 0.f, 
		 5.f, -5.f, 0.f,
		-5.f,  5.f, 0.f,
		 5.f,  5.f, 0.f
	};

	static std::vector<std::pair<ImPlotColormap, tinycolormap::ColormapType>> colormaps =
	{   {ImPlotColormap_Viridis, tinycolormap::ColormapType::Viridis},
		{ImPlotColormap_Plasma, tinycolormap::ColormapType::Plasma},
		{ImPlotColormap_Greys, tinycolormap::ColormapType::Gray},
		{ImPlotColormap_Hot, tinycolormap::ColormapType::Hot},
		{ImPlotColormap_Jet, tinycolormap::ColormapType::Jet} };

	inline std::pair<ImPlotColormap, tinycolormap::ColormapType> getColorMapPair(size_t pos) { return colormaps[pos]; };
}