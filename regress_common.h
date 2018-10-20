#ifndef ECHMET_REGRESS_COMMON_H
#define ECHMET_REGRESS_COMMON_H

#include <array>
#include <vector>
#include <Eigen/Dense>

namespace ECHMET {
namespace RegressCore {

template <typename T> using RegressVector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T> using RegressMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
typedef int msize_t;	/* This must be int in order for OpenMP to build with MSVC */

static const size_t Dynamic = 0;	/*<! Specifies regressor with dynamic number of fitting parameters */

/*
 * Use either fixed size or dynamic size array for data types
 * whose size depends on the number of fitting parameters.
 */
template <typename T, size_t NParams>
class Parametrization {
public:
	typedef typename std::array<T, NParams> ParamsVector;
	typedef typename std::array<bool, NParams> FixedStateVector;
	typedef typename std::array<size_t, NParams> ParamIndexesVector;
};

template <typename T>
class Parametrization<T, Dynamic> {
public:
	typedef typename std::vector<T> ParamsVector;
	typedef typename std::vector<bool> FixedStateVector;
	typedef typename std::vector<size_t> ParamIndexesVector;
};

/*
 * Check that the data type denoting parameter indices makes sense.
 */
template <typename IndexType, size_t NParams>
class ParametrizationValidator {
public:
	static_assert(std::is_enum<IndexType>::value, "IndexType must be an enum for regressors with fixed number of parameters");
};

template <typename IndexType>
class ParametrizationValidator<IndexType, Dynamic> {
public:
	static_assert(std::is_same<IndexType, size_t>::value, "IndexType must be a size_t for regressors with dynamic number of parameters");
};

/*
 * Make sure that only sensible c-tor is available
 * for fixed and dynamic variant of regressor
 */
template <size_t NParams>
class DynamicEnabler {
public:
	typedef void type;
};

template <>
class DynamicEnabler<Dynamic> {
public:
	typedef size_t type;
};

class RegressException : public std::runtime_error {
	using runtime_error::runtime_error;
};

} // namespace RegressCore
} // namespace ECHMET

#endif // ECHMET_REGRESS_COMMON_H


