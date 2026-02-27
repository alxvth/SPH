// SPDX-License-Identifier: MIT
// A corresponding LICENSE.txt file is located in the root directory of this source tree 
// Copyright (C) 2023 Alexander Vieth 

#ifndef EIGEN_PCA_H
#define EIGEN_PCA_H

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace math {

    /// //////// ///
    /// SETTINGS ///
    /// //////// ///

    enum class DATA_NORM {
        NONE,      // no norm 
        MEAN,      // meanNormalization
        MINMAX,    // minMaxNormalization, map each column to [0,1]
    };

    enum class PCA_ALG {
        SVD,    // Use singular value decomposition, Eigen::BDCSVD
        COV,    // Compute eigenvalues of covariance matrix of data, Eigen::SelfAdjointEigenSolver
    };

    /// ////////// ///
    /// CONVERSION ///
    /// ////////// ///

    template<class T>
    inline std::vector<T> convertEigenMatrixToStdVector(Eigen::Matrix<T, -1, -1> mat) {

        const Eigen::StorageOptions StorageOrder = mat.IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor;

        // by default Eigen uses column-major storage order
        if constexpr (StorageOrder == Eigen::ColMajor)
        {
            mat.transposeInPlace();
        }

        return { mat.data(), mat.data() + mat.size() };
    }

    inline Eigen::MatrixXf convertStdVectorToEigenMatrix(const std::vector<float>& data_in, const size_t num_dims)
    {
        const int64_t num_row = data_in.size() / num_dims;
        const int64_t num_col = num_dims;

        if (num_row > std::numeric_limits<int64_t>::max())
            std::cerr << "PCA::convertStdVectorToEigenMatrix can only handle data with up to std::numeric_limits<int64_t>::max() points" << std::endl;

        // convert std vector to Eigen MatrixXf
        // each row in MatrixXf corresponds to one data point
        Eigen::MatrixXf data(num_row, num_col);     	// num_rows (data points), num_cols (attributes)

        // copy data from vector to matrix
#ifndef NDEBUG
#pragma omp parallel for
#endif // NDEBUG
        // loop over data points
        for (int64_t point = 0; point < static_cast<int64_t>(num_row); point++)
        {
            // loop over data point values
            for (int64_t dim = 0; dim < num_col; dim++)
                data(point, dim) = data_in[point * num_dims + dim];
        }

        // this would be more concise but only works if data_in is not const
        // Also, I didn't test this
        //Eigen::MatrixXf data = Eigen::Map<Eigen::MatrixXf>(&data_in[0], num_row, num_col);

        return data;
    }


    /// //// ///
    /// UTIL ///
    /// //// ///

    // Returns the indices that would sort an array, like numpy.argsort
    template<class T, typename C = std::less<>>
    std::vector<size_t> argsort(T& data, C cmp = C{})
    {
        std::vector<size_t> idx(data.size());
        std::iota(idx.begin(), idx.end(), 0);

        std::sort(idx.begin(), idx.end(), [&data, &cmp](const size_t& a, const size_t& b)
            { return cmp(data[a], data[b]); });

        return idx;
    }

    inline Eigen::MatrixXf colwiseZeroMean(const Eigen::MatrixXf& mat) {
        return mat.rowwise() - mat.colwise().mean();
    }

    // Sign correction to ensure deterministic output:
    // flip each dimension such that the max abs value is positive
    // Is similar to svd_flip from scikit-learn, https://github.com/scikit-learn/scikit-learn
    inline Eigen::MatrixXf standardOrientation(const Eigen::MatrixXf& mat)
    {
        // columnswise: which row has the max abs value
        // then get the sign of the max abs value
        Eigen::VectorXf signs(mat.cols());
        Eigen::VectorXf::Index rowID;
        for (uint32_t colID = 0; colID < mat.cols(); colID++)
        {
            mat.col(colID).cwiseAbs().maxCoeff(&rowID);
            signs[colID] = (mat(rowID, colID) >= 0) ? 1.0f : -1.0f;
        }

        // flip columns
        return mat.array().rowwise() * signs.transpose().array();
    }

    inline void _normToCol(const Eigen::VectorXf& normFacs, Eigen::MatrixXf& mat)
    {
        const int32_t num_row = static_cast<int32_t>(mat.rows());
        const int32_t num_col = static_cast<int32_t>(mat.cols());

        // divide all values in mat.col(i) by normFacs[i]
        // return mat_norm.array().rowwise() / normFacs.transpose().array();

        // the previous lines don't seem to work, but the following does
        // there is probably a more elegant way of doing this
        for (int32_t col = 0; col < num_col; col++)
        {
            if (normFacs[col] < 0.0001f) continue;

#ifndef NDEBUG
#pragma omp parallel for
#endif // NDEBUG
            for (int32_t row = 0; row < static_cast<int32_t>(num_row); row++)
            {
                mat(row, col) /= normFacs[col];
            }
        }

    }

    // https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization
    inline Eigen::MatrixXf meanNormalization(const Eigen::MatrixXf& mat)
    {
        // center around mean per attribute
        Eigen::MatrixXf mat_norm = colwiseZeroMean(mat);

        // compute with (max - min) factors
        Eigen::VectorXf normFacs = mat.colwise().maxCoeff() - mat.colwise().minCoeff();

        // norm with (max - min) factors:
        _normToCol(normFacs, mat_norm);

        return mat_norm;
    }

    // map each column to [0,1]
    // https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    inline Eigen::MatrixXf minMaxNormalization(const Eigen::MatrixXf& mat)
    {
        // compute norm factors
        Eigen::VectorXf minVals = mat.colwise().minCoeff();
        Eigen::VectorXf normFacs = mat.colwise().maxCoeff().transpose() - minVals;

        // shift all values
        Eigen::MatrixXf mat_norm = mat.rowwise() - minVals.transpose();

        // norm
        _normToCol(normFacs, mat_norm);

        return mat_norm;
    }


    /// /// ///
    /// PCA ///
    /// /// ///

    // Number pca components must be in [0, std::min(num_row, num_col)]
    static void checkNumComponents(const size_t num_row, const size_t num_col, size_t& num_comp)
    {
        if (num_comp > std::min(num_row, num_col))
        {
            std::cout << "pca: num_comp must be smaller than min(num_row, num_col). Setting num_comp = min(num_row, num_col)" << std::endl;
            num_comp = std::min(num_row, num_col);
        }
        else if (num_comp <= 0)
        {
            std::cout << "pca: num_comp must larger than 0. Setting num_comp = min(num_row, num_col)" << std::endl;
            num_comp = std::min(num_row, num_col);
        }
    }

    // data should be have column-wise zero empirical mean 
    inline Eigen::MatrixXf pcaSVD(const Eigen::MatrixXf& data, const size_t num_comp)
    {
        // compute svd
        Eigen::BDCSVD<Eigen::MatrixXf, Eigen::ComputeThinV> svd(data);

        if (svd.info() != Eigen::Success)
            throw (std::runtime_error("pcaSVD failed. Eigen::ComputationInfo " + std::to_string(static_cast<int32_t>(svd.info()))));

        return svd.matrixV()(Eigen::placeholders::all, Eigen::seq(0, num_comp - 1));
    }

    // data should be have column-wise zero empirical mean 
    inline Eigen::MatrixXf pcaCovMat(const Eigen::MatrixXf& data, const size_t num_comp)
    {
        // covariance matrix
        Eigen::MatrixXf covMat = data.transpose() * data;

        // covariance matrices are symmetric, so use appropriate solver
        Eigen::SelfAdjointEigenSolver <Eigen::MatrixXf> es(covMat);
        Eigen::VectorXf eigenvalues = es.eigenvalues();
        Eigen::MatrixXf eigenvectors = es.eigenvectors();

        if (es.info() != Eigen::Success)
            throw (std::runtime_error("pcaCovMat failed. Eigen::ComputationInfo " + std::to_string(static_cast<int32_t>(es.info()))));

        // sort eigenvalues and save as Eigen::Vector
        auto eigenvalueOrder = argsort(eigenvalues, std::greater{});
        Eigen::Matrix<size_t, -1, 1> eigenvalueOrderE = Eigen::Map<Eigen::Matrix<size_t, -1, 1>, Eigen::Unaligned>(eigenvalueOrder.data(), eigenvalueOrder.size());

        // permutate eigenvalues and -vectors such that the eigenvectors that correspond
        // with the largest eigenvalues are in the first columns
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, size_t> perm(eigenvalueOrderE.transpose());
        eigenvectors = eigenvectors * perm;     // permute columns
        eigenvalues = perm * eigenvalues;

        return eigenvectors(Eigen::placeholders::all, Eigen::seq(0, num_comp - 1));
    }

    inline Eigen::MatrixXf pcaTransform(const Eigen::MatrixXf& data, const Eigen::MatrixXf& principal_components)
    {
        return data * principal_components;
    }

    inline bool pca(const std::vector<float>& data_in, const size_t num_dims, std::vector<float>& pca_out, size_t& num_comp, const PCA_ALG algorithm = PCA_ALG::SVD, const DATA_NORM norm = DATA_NORM::MINMAX, const bool stdOrientation = true)
    {
        // do not transform if data is 1d
        if (num_dims <= 1)
        {
            num_comp = num_dims;
            pca_out = data_in;
            std::cout << "pca: num_dims == 1, no transformation is performed" << std::endl;;
            return false;
        }

        // convert std vector to Eigen MatrixXf
        Eigen::MatrixXf data = convertStdVectorToEigenMatrix(data_in, num_dims);

        // check number of component against number of rows and columns
        const size_t num_row = data.rows();
        const size_t num_col = data.cols();
        size_t _num_comp = num_comp;
        checkNumComponents(num_row, num_col, _num_comp);

        assert(num_row * num_col == data_in.size());
        assert(num_col == num_dims);

        // choose which data normalization to use
        auto norm_data = [&](const Eigen::MatrixXf& dat) {
            if (norm == DATA_NORM::MINMAX)
                return minMaxNormalization(dat);
            else if (norm == DATA_NORM::MEAN)
                return meanNormalization(dat);
            else // norm == DATA_NORM::NONE
                return dat;
            };

        // choose which pcaSVD algorithm to use 
        auto pca_alg = [&](const Eigen::MatrixXf& dat) {
            if (algorithm == PCA_ALG::SVD)
                return pcaSVD(dat, _num_comp);
            else // algorithm == PCA_ALG::COV
                return pcaCovMat(dat, _num_comp);
            };

        // prep data: normalization
        Eigen::MatrixXf data_normed = norm_data(data);

        // Center the values of each variable in the dataset on 0 by subtracting the mean of the variable's observed values from each of those values
        data_normed = colwiseZeroMean(data_normed);

        // compute pcaSVD, get first num_comp components
        Eigen::MatrixXf principal_components;
        try {
            principal_components = pca_alg(data_normed);
        }
        catch (const std::runtime_error& ex) {
            std::cout << "PCA could not be computed: " << ex.what() << std::endl;
            pca_out = std::vector(data.rows() * num_comp, 0.0f);
            return false;
        }

        // project data, compute pca components and
        Eigen::MatrixXf data_transformed = pcaTransform(data_normed, principal_components);

        // enforce same orientation (flip axis) for all algorithms 
        if (stdOrientation)
            data_transformed = math::standardOrientation(data_transformed);

        // convert to std vector with [p0d0, p0d1, ..., p1d0, p1d1, ..., pNd0, pNd1, ..., pNdM]
        pca_out = convertEigenMatrixToStdVector(data_transformed);

        return true;

    }

}

#endif // EIGEN_PCA_H
