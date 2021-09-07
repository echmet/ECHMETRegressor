#ifndef ECHMET_REGRESS_TRACING_H
#define ECHMET_REGRESS_TRACING_H

#include "ECHMETRegressor/tracing/internal/tracer_util.h"
#include "regress_common.h"
#include "tracing/regress_tracer_impl.h"

#ifndef ECHMET_REGRESS_DISABLE_TRACING
#include <sstream>
#include <string>

ECHMET_MAKE_TRACER(RegressTracing)

namespace ECHMET {
namespace RegressCore {

template <typename T>
inline
void DumpIterable(std::stringstream &ss, const T &iterable)
{
	typename T::const_iterator cit = iterable.cbegin();

	while (cit != iterable.cend()) {
		ss << *cit << "\n";
		cit++;
	}
}

template <typename T>
inline
void DumpMatrix(std::stringstream &ss, const RegressMatrix<T> &m)
{
	ss << "[" << m.rows() << "] [" << m.cols() << "]" << "\n";
	ss << m << "\n";
}

template <typename T>
inline
void DumpVector(std::stringstream &ss, const RegressVector<T> &v)
{
	ss << "[" << v.size() << "]" << "\n";
	ss << v << "\n";
}

inline
std::string MakeLogTitle(const std::string &title)
{
	return "--- " + title + " ---\n";
}

template <typename T>
inline
std::stringstream LogMatrix(const std::string &title, const RegressMatrix<T> &m)
{
	std::stringstream ss{};

	ss << MakeLogTitle(title);
	DumpMatrix(ss, m);

	ss << "\n";

	return ss;
}

template <typename T>
inline
std::stringstream LogVector(const std::string &title, const RegressVector<T> &m)
{
	std::stringstream ss{};

	ss << MakeLogTitle(title);
	DumpVector(ss, m);

	ss << "\n";

	return ss;
}

template <typename T>
inline
std::stringstream LogIterable(const std::string &title, const T &iterable)
{
	std::stringstream ss{};

	ss << MakeLogTitle(title);
	DumpIterable(ss, iterable);

	ss << "\n";

	return ss;
}

} // namespace RegressCore

ECHMET_MAKE_TRACEPOINT(RegressTracing, MATRIX_ALPHA, "Matrix alpha")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, MATRIX_ALPHA, const RegressCore::RegressMatrix<T1> &m)
{
	return RegressCore::LogMatrix("Matrix alpha", m).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, MATRIX_BETA, "Matrix beta")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, MATRIX_BETA, const RegressCore::RegressMatrix<T1> &m)
{
	return RegressCore::LogMatrix("Matrix beta", m).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, MATRIX_DELTA, "Matrix delta")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, MATRIX_DELTA, const RegressCore::RegressMatrix<T1> &m)
{
	return RegressCore::LogMatrix("Matrix delta", m).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, REGRESS_STEP, "Regress step")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, REGRESS_STEP, const T1 &sOld, const T1 &sNew, const T1 &sAccepted, const T1 &dS, const T1 &improvement, const T1 &sInitial)
{
	std::stringstream ss{};

	ss << RegressCore::MakeLogTitle("Regress step");

	ss << "S old: " << sOld << "\n"
           << "S new: " << sNew << "\n"
	   << "S accepted: " << sAccepted << "\n"
	   << "delta S: " << dS << "\n"
	   << "Improvement: " << improvement << "\n"
	   << "S initial: " << sInitial;

	ss << "\n\n";

	return ss.str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, RESULT_ACCEPTANCE, "Result acceptance")
ECHMET_BEGIN_MAKE_LOGGER(RegressTracing, RESULT_ACCEPTANCE, const bool accepted)
{
	std::string s = RegressCore::MakeLogTitle("Result acceptance");

	if (accepted)
		return s + "ACCEPTED\n\n";
	return s + "NOT ACCEPTED\n\n";
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, MARK_STEP_BOUNDS, "Mark step bounds")
ECHMET_BEGIN_MAKE_LOGGER_NOARGS(RegressTracing, MARK_STEP_BOUNDS)
{
	return "\n==== ==== ====\n\n";
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, MARK_REGRESS_RESTART, "Mark regressor restart")
ECHMET_BEGIN_MAKE_LOGGER_NOARGS(RegressTracing, MARK_REGRESS_RESTART)
{
	return "\n==== RESTART ====\n\n";
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, LAMBDAS, "Lambdas")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, LAMBDAS, const T1 &lambda, const T1 &lambdaCoeff)
{
	std::stringstream ss{};

	ss << RegressCore::MakeLogTitle("Lambdas");

	ss << "Lambda: " << lambda << "\n"
           << "Lambda coeff: " << lambdaCoeff << "\n";

	ss << "\n\n";

	return ss.str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, VECTOR_X, "Vector of X values to operate on")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, VECTOR_X, const std::vector<T1> &v)
{
	return RegressCore::LogIterable("Vector X", v).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, VECTOR_Y, "Vector of Y values for given X values")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, VECTOR_Y, const RegressCore::RegressVector<T1> &v)
{
	return RegressCore::LogVector("Vector Y", v).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, VECTOR_PARAMS, "Vector of parameters")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, VECTOR_PARAMS, const T1 &params)
{
	return RegressCore::LogIterable("Vector of parameters", params).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, EVALUATION_ERRORS, "Simple reporting of evaluation errors")
ECHMET_BEGIN_MAKE_LOGGER(RegressTracing, EVALUATION_ERRORS, const std::string &error)
{
	return RegressCore::MakeLogTitle("Evaluation error") + error + "\n\n";
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, MATRIX_P, "Matrix of derivatives")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, MATRIX_P, const RegressCore::RegressMatrix<T1> &m)
{
	return RegressCore::LogMatrix("Matrix of derivatives", m).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, VECTOR_FX, "Vector Fx")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, VECTOR_FX, const RegressCore::RegressVector<T1> &v)
{
	return RegressCore::LogVector("Vector Fx", v).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, VECTOR_ERROR, "Vector of errors")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, VECTOR_ERROR, const RegressCore::RegressVector<T1> &v)
{
	return RegressCore::LogVector("Vector of errors", v).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, MATRIX_ALPHA_FINAL, "Final alpha atrix")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, MATRIX_ALPHA_FINAL, const RegressCore::RegressMatrix<T1> &m)
{
	return RegressCore::LogMatrix("Matrix alpha - final", m).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, MATRIX_COVARIANCE, "Final alpha matrix")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, MATRIX_COVARIANCE, const RegressCore::RegressMatrix<T1> &m)
{
	return RegressCore::LogMatrix("Covariance matrix", m).str();
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, ZERO_DF_MODE, "Zero degrees-of-freedom mode")
ECHMET_BEGIN_MAKE_LOGGER(RegressTracing, ZERO_DF_MODE, const bool zeroDFmode)
{
	if (zeroDFmode)
		return "Using zero degrees-of-freedom mode";
	return "Using normal degrees-of-freedom-mode";
}
ECHMET_END_MAKE_LOGGER

ECHMET_MAKE_TRACEPOINT(RegressTracing, SANITY_CHECK_FAILURE, "Parameters sanity check failure")
ECHMET_BEGIN_MAKE_LOGGER_T1(RegressTracing, SANITY_CHECK_FAILURE, const T1 &m)
{
	std::stringstream ss{};

	ss << RegressCore::MakeLogTitle("Parameters failed sanity check");
	ss << RegressCore::LogIterable("Vector of parameters", m).str();
	ss << "\n\n";

	return ss.str();
}
ECHMET_END_MAKE_LOGGER

} // namespace ECHMET

#else // ECHMET_REGRESS_DISABLE_TRACING

ECHMET_MAKE_TRACER(ECHMET::__DUMMY_TRACER_CLASS)

#endif // ECHMET_REGRESS_DISABLE_TRACING

#endif // ECHMET_REGRESS_TRACING_H
