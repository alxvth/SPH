#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "CommonDefinitions.hpp"
#include "ProgressBar.hpp"

#include <ankerl/unordered_dense.h>
#include <hdi/dimensionality_reduction/hd_joint_probability_generator.h>

namespace sph::utils {

    /// ////////////// ///
    /// t-SNE (HDILib) ///
    /// ////////////// ///

    // as in hdi/utils/math_utils.h but with a extra static casts
    template <typename Vector>
    double computeGaussianDistributionWithFixedPerplexity(typename Vector::const_iterator distances_begin, typename Vector::const_iterator distances_end, typename Vector::iterator distribution_begin, typename Vector::iterator distribution_end, double perplexity, int max_iterations, double tol, int ignore) {
        const int size(static_cast<int>(std::distance(distances_begin, distances_end)));
        if (size != std::distance(distribution_begin, distribution_end) || size == 0) {
            throw std::logic_error("Invalid containers");
        }

        bool found      = false;
        double beta     = 1.;
        double sigma    = std::sqrt(1 / (2 * beta));
        double min_beta = -std::numeric_limits<double>::max();
        double max_beta = std::numeric_limits<double>::max();

        constexpr double double_max = std::numeric_limits<double>::max();

        // Iterate until we found a good perplexity
        int iter = 0;
        double sum_distribution = std::numeric_limits<double>::min();
        while (!found && iter < max_iterations) {
            // Compute Gaussian kernel row
            sum_distribution = std::numeric_limits<double>::min();
            {
                auto distance_iter = distances_begin;
                auto distribution_iter = distribution_begin;
                for (int idx = 0; distance_iter != distances_end; ++distance_iter, ++distribution_iter, ++idx) {
                    if (idx == ignore) {
                        (*distribution_iter) = 0;
                        continue;
                    }
                    double v = exp(-beta * static_cast<double>(*distance_iter));
                    sigma = std::sqrt(1 / (2 * beta));
                    //double v = exp(- (*distance_iter) / (2*sigma*sigma));
                    (*distribution_iter) = static_cast<Vector::value_type>(v);
                    sum_distribution += v;
                }

            }

            double H = .0; //entropy
            {
                auto distance_iter = distances_begin;
                auto distribution_iter = distribution_begin;
                for (int idx = 0; distance_iter != distances_end; ++distance_iter, ++distribution_iter, ++idx) {
                    if (idx == ignore)
                        continue;
                    H += beta * ((*distance_iter) * (*distribution_iter));
                }
                H = (H / sum_distribution) + log(sum_distribution);
            }


            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if (Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if (Hdiff > 0) {
                    min_beta = beta;
                    if (max_beta == double_max || max_beta == -double_max)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if (min_beta == -double_max || min_beta == double_max)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }
            iter++;

        }
        if (!found) {
            auto v = static_cast<Vector::value_type>(1. / (size + ((ignore < 0 || ignore >= size) ? 0 : -1)));
            for (auto distribution_iter = distribution_begin; distribution_iter != distribution_end; ++distribution_iter) {
                (*distribution_iter) = v;
            }
            return 0;
        }
        for (auto distribution_iter = distribution_begin; distribution_iter != distribution_end; ++distribution_iter) {
            (*distribution_iter) = static_cast<Vector::value_type>(*distribution_iter / sum_distribution);
        }
        return sigma;
    }

    // same as in hdi/dimensionality_reduction/hd_joint_probability_generator.h but with a extra static casts and stand-alone function
    template <typename scalar, typename sparse_scalar_matrix, typename integer>
    void computeGaussianDistributions(const std::vector<scalar>& distances_squared, const std::vector<integer>& indices, int nn, sparse_scalar_matrix& distribution, typename ::hdi::dr::HDJointProbabilityGenerator<scalar, sparse_scalar_matrix>::Parameters& params) {
        const size_t n = distribution.size();
        std::vector<scalar> temp_vector(distances_squared.size(), 0);

        SPH_PARALLEL
        for (int64_t j = 0; j < static_cast<int64_t>(n); ++j) {
            [[maybe_unused]] const auto sigma = utils::computeGaussianDistributionWithFixedPerplexity<std::vector<scalar>>(
                distances_squared.begin() + j * nn, //check squared
                distances_squared.begin() + (j + 1) * nn,
                temp_vector.begin() + j * nn,
                temp_vector.begin() + (j + 1) * nn,
                params._perplexity,
                200,
                1e-5,
                0
            );
        }

        for (size_t j = 0; j < n; ++j) {
            for (int k = 1; k < nn; ++k) {
                const size_t i = j * nn + k;
                // if items do not have all the same number of neighbors this is indicated by -1
                if (indices[i] == -1)
                    continue;
                distribution[j][static_cast<sparse_scalar_matrix::value_type::key_type>(indices[i])] = temp_vector[i];
            }
        }
    }

    // as in hdi/utils/graph_algorithms.h but with a second template parameter uinteger
    template <class map_type, typename uinteger>
    void extractSubGraph(const std::vector<map_type>& orig_transition_matrix, const std::vector<uinteger>& selected_idxes, std::vector<map_type>& new_transition_matrix, std::vector<uinteger>& new_idxes, typename map_type::mapped_type thresh)
    {
        using hash = ankerl::unordered_dense::hash<uinteger>;
        using hashmap = ankerl::unordered_dense::map<uinteger, uinteger, hash>;

        new_transition_matrix.clear();
        new_idxes.clear();
        new_idxes.reserve(selected_idxes.size());

        hashmap map_selected_idxes;
        hashmap map_non_selected_idxes;

        // The selected rows must be taken completely
        for (const auto id : selected_idxes) {
            map_selected_idxes[id] = new_idxes.size();
            new_idxes.push_back(id);
        }

        // Vertices that are connected to a selected vertex
        for (const auto& e : map_selected_idxes) {
            for (const auto& row_elem : orig_transition_matrix[e.first]) {
                if (row_elem.second > thresh) {
                    if (map_selected_idxes.find(row_elem.first) == map_selected_idxes.end() &&
                        map_non_selected_idxes.find(row_elem.first) == map_non_selected_idxes.end()) {
                        map_non_selected_idxes[row_elem.first] = new_idxes.size();
                        new_idxes.push_back(row_elem.first);
                    }
                }
            }
        }
        // Now that I have the maps, I generate the new transition matrix
        new_transition_matrix.resize(map_non_selected_idxes.size() + map_selected_idxes.size());
        for (const auto& e : map_selected_idxes) {
            for (const auto row_elem : orig_transition_matrix[e.first]) {
                if (map_selected_idxes.find(row_elem.first) != map_selected_idxes.end()) {
                    new_transition_matrix[e.second][map_selected_idxes[row_elem.first]] = row_elem.second;
                }
                else if (map_non_selected_idxes.find(row_elem.first) != map_non_selected_idxes.end()) {
                    new_transition_matrix[e.second][map_non_selected_idxes[row_elem.first]] = row_elem.second;
                }
            }
        }
        for (const auto& e : map_non_selected_idxes) {
            for (const auto row_elem : orig_transition_matrix[e.first]) {
                if (map_selected_idxes.find(row_elem.first) != map_selected_idxes.end()) {
                    new_transition_matrix[e.second][map_selected_idxes[row_elem.first]] = row_elem.second;
                }
                else if (map_non_selected_idxes.find(row_elem.first) != map_non_selected_idxes.end()) {
                    new_transition_matrix[e.second][map_non_selected_idxes[row_elem.first]] = row_elem.second;
                }
            }
        }

        // Finally, the new transition matrix must be normalized
        double sum = 0;
        for (const auto& row : new_transition_matrix) {
            for (const auto& elem : row) {
                sum += elem.second;
            }
        }
        for (auto& row : new_transition_matrix) {
            for (auto& elem : row) {
                elem.second = new_transition_matrix.size() * elem.second / sum;
            }
        }

    }

    // as in hdi/utils/graph_algorithms.h but with a second template parameter uinteger
    // optionally does not extract connected vertices, equivalent to extractSubGraph with thresh = 1
    // thus new_idxes == selected_idxes
    template <class map_type, typename uinteger>
    void extractSubGraph(const std::vector<map_type>& orig_transition_matrix, const std::vector<uinteger>& selected_idxes, std::vector<map_type>& new_transition_matrix, std::vector<uinteger>& new_idxes)
    {
        using hash = ankerl::unordered_dense::hash<uinteger>;
        using hashmap = ankerl::unordered_dense::map<uinteger, uinteger, hash>;

        new_transition_matrix.clear();
        new_idxes.clear();
        new_idxes.reserve(selected_idxes.size());

        hashmap map_selected_idxes;

        // The selected rows must be taken completely
        for (const auto id : selected_idxes) {
            map_selected_idxes[id] = new_idxes.size();
            new_idxes.push_back(id);
        }

        // Now that I have the maps, I generate the new transition matrix
        new_transition_matrix.resize(map_selected_idxes.size());
        for (const auto& e : map_selected_idxes) {
            for (const auto row_elem : orig_transition_matrix[e.first]) {
                if (map_selected_idxes.find(row_elem.first) != map_selected_idxes.end()) {
                    new_transition_matrix[e.second][map_selected_idxes[row_elem.first]] = row_elem.second;
                }
            }
        }

        // Finally, the new transition matrix must be normalized
        double sum = 0;
        for (const auto& row : new_transition_matrix) {
            for (const auto& elem : row) {
                sum += elem.second;
            }
        }
        for (auto& row : new_transition_matrix) {
            for (auto& elem : row) {
                elem.second = new_transition_matrix.size() * elem.second / sum;
            }
        }

    }

    // as in hdi/dimensionality_reduction/HDJointProbabilityGenerator_inl.h -> symmetrize(), but not private 
    // skip double compute and fewer binary searches
    template <class map_type = sph::SparseVecHDI, typename uinteger = uint32_t>
    void symmetrizeTSNE(std::vector<map_type>& distribution) {
        size_t numPoints = distribution.size();
        utils::ProgressBar progress(numPoints);
        for (uinteger j = 0; j < static_cast<uinteger>(numPoints); ++j) {
            for (auto& e : distribution[j]) {
                const auto i = e.first;
                auto& val_ji = e.second;
                auto& val_ij = distribution[i][j];

                if (j > i && val_ij > 0.f)
                    continue;

                const auto new_val = (val_ji + val_ij) * 0.5f;
                val_ji = new_val;
                val_ij = new_val;
            }
            progress.update();
        }
        progress.finish();
    }

    template <class map_type = sph::SparseVecHDI, typename uinteger = uint32_t>
    void symmetrizeUMAP(std::vector<map_type>& distribution) {
        size_t numPoints = distribution.size();
        utils::ProgressBar progress(numPoints);
        for (uinteger j = 0; j < static_cast<uinteger>(numPoints); ++j) {
            for (auto& e : distribution[j]) {
                const auto i = e.first;
                auto& val_ji = e.second;
                auto& val_ij = distribution[i][j];

                if (j > i && val_ij > 0.f)
                    continue;

                const auto new_val = (val_ij + val_ji) - (val_ij * val_ji);
                val_ji = new_val;
                val_ij = new_val;
            }
            progress.update();
        }
        progress.finish();
    }

} // namespace sph::utils

