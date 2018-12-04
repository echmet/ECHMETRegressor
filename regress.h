#ifndef ECHMET_REGRESS_H
#define ECHMET_REGRESS_H

#include "regress_tracing.h"

#include <cmath>
#include <limits>
#include <new>

namespace ECHMET {

namespace RegressCore {

/*!
 * Abstract regressor class.
 *
 * @tparam XT XT value type.
 * @tparam YT YT value type.
 * @tparam NParams Number of fitting parameters. Set to <tt>Dynamic</tt> for variable number of parameters.
 * @tparam IndexType Value type of fitting parameters list. This must be <tt>size_t</tt> for dynamic regressors
 *                   and an enum for regressors with fixed number of parameters.
 */
template <typename XT = double, typename YT = double, size_t NParams = Dynamic, typename IndexType = size_t>
class RegressFunction {
public:
	static_assert(NParams < std::numeric_limits<msize_t>::max(), "Maximum number of fitting parameters exceeded");

	typedef XT x_type;
	typedef YT y_type;
	typedef IndexType index_type;

	typedef RegressMatrix<YT> YTMatrix;
	typedef std::vector<XT> XTVector;
	typedef RegressVector<YT> YTVector;

	typedef typename Parametrization<YT, NParams>::ParamsVector ParamsVector;
	typedef typename Parametrization<YT, NParams>::FixedStateVector FixedStateVector;
	typedef typename Parametrization<YT, NParams>::ParamIndexesVector ParamIndexesVector;
	typedef typename ParamsVector::size_type PVSize;

	typedef void (*report_function)(RegressFunction const &, void *);

	template <typename SZType = size_t, typename = typename std::enable_if<std::is_same<SZType, typename DynamicEnabler<NParams>::type>::value>::type>
	explicit RegressFunction(const size_t params) :
		m_params(params),
		m_notFixed(params),
		m_report_function(nullptr),
		m_report_function_context(nullptr),
		m_rss(0),
		m_nmax(0),
		m_epsilon(1E-9),
		m_damping(true)
	{
		ParametrizationValidator<IndexType, NParams>();

		Reset();

		m_fixedParams = FixedStateVector(params, false);
		m_pindexes.resize(params);
	}

	template <typename SZType = size_t, typename = typename std::enable_if<!std::is_same<SZType, typename DynamicEnabler<NParams>::type>::value>::type>
	explicit RegressFunction() :
		m_notFixed(NParams),
		m_report_function(nullptr),
		m_report_function_context(nullptr),
		m_rss(0),
		m_nmax(0),
		m_epsilon(1E-9),
		m_damping(true)
	{
		ParametrizationValidator<IndexType, NParams>();

		Reset();
	}

	RegressFunction(const RegressFunction &) = delete;
	virtual ~RegressFunction();

	void operator=(const RegressFunction &) = delete;

	void Abort();
	void Assign(const RegressFunction &other);
	RegressFunction * Clone() const;
	RegressFunction * Clone(std::nothrow_t) const;
	void FixParameter(const IndexType id);
	void FixParameter(const IndexType id, const YT &val);
	bool Initialize(const XTVector &x, const YTVector &y, const YT &eps, int nmax, bool damp);
	bool IsFixedParameter(const IndexType id);
	bool Regress();
	void RegisterReportFunction(report_function, void *);
	void ReleaseParameter(const IndexType id);
	void Report() const;
	void UnregisterReportFunction();

	YTVector GetVariances() const
	{
		YTVector variances(m_params.size());

		typename YTMatrix::Index j = 0;
		for (typename YTVector::Index i = 0; i < variances.size(); i++) {
			if (!m_fixedParams[i]) {
				variances(i) = m_covariance(j, j);
				j++;
			} else
				variances(i) = 0.0;
		}

		return variances;
	}

	report_function GetReportFunction() const
	{
		return m_report_function;
	}

	const ParamsVector & GetParameters() const
	{
		return m_params;
	}

	const XTVector & GetXs() const
	{
		return m_x;
	}

	const YTMatrix & GetYs() const
	{
		return m_y;
	}

	const YTVector & GetFx() const
	{
		return m_fx;
	}

	const YTVector & GetErrors() const
	{
		return m_error;
	}

	YT GetRSS() const
	{
		return m_rss;
	}

	YT GetImprovement() const
	{
		return m_improvement;
	}

	int GetIterationCounter() const
	{
		return m_iterationCounter;
	}

	msize_t GetNMax() const
	{
		return m_nmax;
	}

	YT GetEps() const
	{
		return m_epsilon;
	}

	bool IsDamped() const
	{
		return m_damping;
	}

	bool IsAborted () const
	{
		return m_aborted;
	}

	bool IsAccepted() const
	{
		return m_accepted;
	}

	bool HasConverged() const;

	size_t GetNCount () const;
	size_t GetPCount () const; // 1
	size_t GetPTotal () const
	{
		return this->m_params.size();
	}

	size_t GetDF () const;
	YT GetMSE () const;
	YT GetS2 () const;
	YT GetS () const;

	//------------------------------------------------------------------------
	// 1) Number of not fixed(!) parameters (those to be estimated)
	// 2) Total number of parameters (both fixed and not fixed)

	YT Evaluate(const XT &x) const;
	YT operator()(const XT &x) const;

protected:
	//---
	// Initialized in Initialize
	XTVector m_x;   //data x
	YTVector m_fx;  //fx calculated

	// Initialized in constructor
	ParamsVector m_params;

	// Initialized in OnParamsChanged
	YTMatrix m_p;  //derivation matrix        [not_fixed, x]

	// Non-resetable state variables
	size_t m_notFixed;

	/*!
	 * Optional sanity check of the estimated parameters.
	 *
	 * @param params Parameters to validate
	 *
	 * @return true if the parameters are sane, false otherwise.
	 */
	virtual bool ACheckSanity(const ParamsVector &params) const
	{
		(void)params;

		return true;
	}

	virtual RegressFunction * ACreate() const = 0;
	virtual void AAssign(const RegressFunction &other) = 0;
	virtual YT ACalculateFx(const XT &x, const ParamsVector &params, msize_t idx) const = 0;
	virtual YT ACalculateDerivative(const XT &x, const ParamsVector &params, const IndexType param_idx, msize_t idx) const = 0;
	virtual void ACalculateP();
	virtual bool ASetInitialParameters(ParamsVector &params, const XTVector &x, const YTVector &y) = 0;
	void CheckMatrix(YTMatrix const &matrix) noexcept(false);
	const YT & GetParam(const ParamsVector &params, const IndexType id) const;
	void SetParam(ParamsVector &params, const IndexType id, const YT &value);

private:
	report_function m_report_function;
	void * m_report_function_context;

	// Initialized in constructor
	FixedStateVector m_fixedParams;     //           [params]
	ParamIndexesVector m_pindexes;        //           [params]

	// Initialized in Initialize
	YTVector m_y;   //data y                 [x]

	// Initialized in CalculateRSS
	YTVector m_error;      //                         [x]

	// Initialized in Regress
	YTMatrix m_alpha;      //                         [not_fixed, not_fixed]
	YTVector m_beta;       //                         [not_fixed]
	YTVector m_delta;      //                         [not_fixed]
	YTMatrix m_covariance; //

	// Resetable state variables

	int m_iterationCounter;

	bool m_accepted;
	volatile bool m_aborted;

	YT m_improvement;
	YT m_lambda;
	YT m_lambdaCoeff;

	// Non-resetable state variables
	YT m_rss;

	// Setup variables
	int m_nmax;
	YT m_epsilon;
	bool m_damping;

	void Reset();

	void OnParamsChanged(const bool regressCall = false);
	virtual void OnParamsChangedInternal();

	void CalculateFx();
	void CalculateRSS();

	void CheckSolution(const YTMatrix &A, const YTMatrix &result, const YTMatrix &rhs) const noexcept(false);

	const YT & GetParamInternal(const ParamsVector &params, const PVSize idx) const;
	void FixParameterInternal(const PVSize idx);
	void FixParameterInternal(const PVSize idx, const YT &val);
	bool IsFixedParameterInternal(const PVSize idx);
	void ReleaseParameterInternal(const PVSize idx);
	void SetParamInternal(ParamsVector &params, const PVSize idx, const YT &value);
};

template <typename XT, typename YT, size_t NParams, typename IndexType>
RegressFunction<XT, YT, NParams, IndexType>::~RegressFunction()
{}

template <typename XT, typename YT, size_t NParams, typename IndexType>
bool RegressFunction<XT, YT, NParams, IndexType>::Initialize(const XTVector &x, const YTVector & y, const YT &eps, int nmax, bool damp)
{
	if (x.size() > std::numeric_limits<msize_t>::max())
		return false;

	if (x.size() == 0 || static_cast<msize_t>(x.size()) != y.size())
		return false;

	Reset();
	//-------------------------------------------------------------------------
	// Call again in OnParamsChanged but we may return
	// before even getting there

	m_rss = 0;

	m_nmax = nmax;
	m_epsilon  = eps;
	m_damping  = damp;

	m_x = x;
	m_y = y;

	if (!ASetInitialParameters(m_params, m_x, m_y))
		return false;

	m_fx = YTVector(m_y.rows());

	OnParamsChanged();

	return true;
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
RegressFunction<XT, YT, NParams, IndexType> * RegressFunction<XT, YT, NParams, IndexType>::Clone() const
{
	RegressFunction * result = this->ACreate();
	result->Assign(*this);

	return result;
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
RegressFunction<XT, YT, NParams, IndexType> * RegressFunction<XT, YT, NParams, IndexType>::Clone(std::nothrow_t) const
{
	try {
		return Clone();
	} catch(std::bad_alloc &) {
		return nullptr;
	} catch (std::bad_cast &) {
		return nullptr;
	}
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::Assign(const RegressFunction &other)
{
	m_report_function = other.m_report_function;

	m_params = other.m_params;
	m_fixedParams = other.m_fixedParams;

	m_x = other.m_x;
	m_y = other.m_y;
	m_fx = other.m_fx;

	m_p = other.m_p;

	m_error = other.m_error;

	m_alpha = other.m_alpha;
	m_beta = other.m_beta;
	m_delta = other.m_delta;

	m_iterationCounter = other.m_iterationCounter;

	m_accepted = other.m_accepted;
	m_aborted = other.m_aborted;

	m_improvement = other.m_improvement;
	m_lambda = other.m_lambda;
	m_lambdaCoeff = other.m_lambdaCoeff;

	m_rss = other.m_rss;
	m_notFixed = other.m_notFixed;

	m_nmax = other.m_nmax;
	m_epsilon = other.m_epsilon;
	m_damping = other.m_damping;

	AAssign(other);
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
bool RegressFunction<XT, YT, NParams, IndexType>::Regress()
{
	// INIT

	Reset();

RESTART:
	if (m_notFixed != 0)
		m_improvement = -2. * m_epsilon; // 1)
	else
		m_accepted = true;
	//-------------------------------------------------
	// 1) anything so that abs(m_improvement) > m_epsilon
	//    positive -> m_lambda *= m_lambdaCoeff at first call
	//    negative -> m_lambda /= m_lambdaCoeff at first call

	m_lambda = 1. / 16384.;

	YT sAccepted = this->GetS();
	ParamsVector paramsAccepted(m_params);
	const bool zeroDFMode = this->GetDF() == 0;
	const auto continueFit = [this, zeroDFMode]() {
		if (zeroDFMode)
			return !m_accepted;
		return std::abs(m_improvement) > m_epsilon;
	};

	ECHMET_TRACE(RegressTracing, ZERO_DF_MODE, zeroDFMode);

	// DOIT

	while (m_iterationCounter != m_nmax && continueFit() && !m_aborted) {
		m_alpha = m_p * m_p.transpose();

		if (m_damping) {
			if (m_improvement < 0)
				m_lambda *= m_lambdaCoeff;
			else
				m_lambda /= m_lambdaCoeff;

			ECHMET_TRACE_T1(RegressTracing, LAMBDAS, YT, m_lambda, m_lambdaCoeff);

			m_alpha.diagonal(0) *= (1 + m_lambda);
		}

		ECHMET_TRACE_T1(RegressTracing, MATRIX_ALPHA, YT, m_alpha);

		m_beta = m_p * m_error;

		ECHMET_TRACE_T1(RegressTracing, MATRIX_BETA, YT, m_beta);

		try {
			CheckMatrix(m_alpha);

			m_delta = m_alpha.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(m_beta);
			CheckSolution(m_alpha, m_delta, m_beta);
		} catch (const RegressException &) {
			ECHMET_TRACE(RegressTracing, EVALUATION_ERRORS, "Singularity error");

			m_accepted = false;
			m_iterationCounter++;
			goto FINALIZE;
	        }

		ECHMET_TRACE_T1(RegressTracing, MATRIX_DELTA, YT, m_delta);

		for (typename FixedStateVector::size_type i = 0, j = 0; i < m_fixedParams.size(); ++i) {
			if (!m_fixedParams[i]) {
				m_params[i] += m_delta(j);
				++j;
			}
		}

		++m_iterationCounter;
		if (!ACheckSanity(m_params)) {
			m_accepted = false;
			goto FINALIZE;
		}

		if (!zeroDFMode) {
			YT sOld = this->GetS();
			OnParamsChanged(true);
			YT sNew = this->GetS();
			m_improvement = sOld - sNew;
			m_accepted = sNew < sAccepted + m_epsilon;

			ECHMET_TRACE_T1(RegressTracing, REGRESS_STEP, YT, sOld, sNew, sAccepted, sAccepted - sNew, m_improvement);

			if (m_accepted) {
				paramsAccepted = m_params;
				m_improvement = sAccepted - sNew;
				sAccepted = sNew;
			}
		} else {
			OnParamsChanged(true);
			m_accepted = true;
			for (int idx = 0; idx < m_error.size(); idx++) {
				if (std::abs(m_error(idx)) > m_epsilon)
					m_accepted = false;
			}

			if (m_accepted)
				paramsAccepted = m_params;
		}

		if (m_accepted) {
			ECHMET_TRACE(RegressTracing, RESULT_ACCEPTANCE, true);

		} else {
			/* ECHMET_TRACE translates to nothing if tracing is disabled,
			 * leave the enclosing braces here! */
			ECHMET_TRACE(RegressTracing, RESULT_ACCEPTANCE, false);
		}

		Report();

		ECHMET_TRACE_NOARGS(RegressTracing, MARK_STEP_BOUNDS);
	}

FINALIZE:

	m_params = paramsAccepted;
	OnParamsChanged(true);

	if (!m_accepted && m_iterationCounter != m_nmax && !m_aborted) {
		ECHMET_TRACE_NOARGS(RegressTracing, MARK_REGRESS_RESTART);

		m_lambdaCoeff *= 2;

		ECHMET_TRACE_T1(RegressTracing, LAMBDAS, YT, m_lambda, m_lambdaCoeff);

		goto RESTART;
	} else {
		/* We have done what we could, calculate covariance matrix */
		m_covariance = m_p * m_p.transpose();

		ECHMET_TRACE_T1(RegressTracing, MATRIX_ALPHA_FINAL, YT, std::cref(m_covariance));

		m_covariance = m_covariance.inverse();
		ECHMET_TRACE_T1(RegressTracing, MATRIX_COVARIANCE, YT, std::cref(m_covariance));
	}

	Report();

	return HasConverged();
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::Abort()
{
	m_aborted = true;
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::Report() const
{
	if (m_report_function != nullptr)
		m_report_function(*this, m_report_function_context);
}

//---------------------------------------------------------------------------
template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::RegisterReportFunction(report_function f, void *context)
{
	m_report_function = f;
	m_report_function_context = context;
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::UnregisterReportFunction()
{
	m_report_function = nullptr;
	m_report_function_context = nullptr;
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::FixParameter(const IndexType id)
{
	const auto idx = static_cast<PVSize>(id);

	FixParameterInternal(idx);
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::FixParameterInternal(const PVSize idx)
{
	if (IsFixedParameter(idx))
		return;

	assert(idx >= 0 && idx < m_fixedParams.size());

	m_fixedParams[idx] = true;
	--m_notFixed;

	OnParamsChanged();
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::FixParameter(const IndexType id, const YT &val)
{
	const auto idx = static_cast<PVSize>(id);

	FixParameterInternal(idx, val);
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::FixParameterInternal(const PVSize idx, const YT &val)
{
	assert(idx >= 0 && idx < m_params.size());

	m_params[idx] = val;

	if (!IsFixedParameter(idx))
		FixParameterInternal(idx);
	else
		OnParamsChanged();
}

//---------------------------------------------------------------------------
template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::ReleaseParameter(const IndexType id)
{
	const auto idx = static_cast<PVSize>(id);

	ReleaseParameterInternal(idx);
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::ReleaseParameterInternal(const PVSize idx)
{
	if (!IsFixedParameterInternal(idx))
		return;

	assert(idx >= 0 && idx < m_fixedParams.size());

	m_fixedParams[idx] = false;
	++m_notFixed;

	OnParamsChanged();
}

//---------------------------------------------------------------------------
template <typename XT, typename YT, size_t NParams, typename IndexType>
bool RegressFunction<XT, YT, NParams, IndexType>::IsFixedParameter(const IndexType id)
{
	const auto idx = static_cast<typename FixedStateVector::size_type>(id);
	assert(idx >= 0 && idx < m_fixedParams.size());

	return m_fixedParams[idx];
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
bool RegressFunction<XT, YT, NParams, IndexType>::IsFixedParameterInternal(const PVSize idx)
{
	assert(idx >= 0 && idx < m_fixedParams.size());

	return m_fixedParams[idx];
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
bool RegressFunction<XT, YT, NParams, IndexType>::HasConverged() const
{
	if (this->GetDF() == 0)
		return m_accepted;
	return m_accepted && std::abs(m_improvement) < m_epsilon;
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
size_t RegressFunction<XT, YT, NParams, IndexType>::GetNCount() const
{
	return m_y.rows();
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
size_t RegressFunction<XT, YT, NParams, IndexType>::GetPCount() const
{
	return m_notFixed;
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
size_t RegressFunction<XT, YT, NParams, IndexType>::GetDF() const
{
	return GetNCount() - GetPCount();
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
YT RegressFunction<XT, YT, NParams, IndexType>::GetMSE() const
{
	return GetRSS() / GetNCount();
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
YT RegressFunction<XT, YT, NParams, IndexType>::GetS2() const
{
	return GetRSS() / GetDF();
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
YT RegressFunction<XT, YT, NParams, IndexType>::GetS() const
{
	return ::std::sqrt(GetS2());
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
YT RegressFunction<XT, YT, NParams, IndexType>::Evaluate(const XT &x) const
{
	return ACalculateFx(x, m_params, -1);
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
YT RegressFunction<XT, YT, NParams, IndexType>::operator()(const XT &x) const
{
	return Evaluate(x);
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
const YT & RegressFunction<XT, YT, NParams, IndexType>::GetParam(const ParamsVector &params, const IndexType id) const
{
	const auto idx = static_cast<typename ParamsVector::size_type>(id);

	return GetParamInternal(params, idx);
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
const YT & RegressFunction<XT, YT, NParams, IndexType>::GetParamInternal(const ParamsVector &params, const PVSize idx) const
{
	assert(idx >= 0 && idx < params.size());

	return params[idx];
}

//---------------------------------------------------------------------------
template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::SetParam(ParamsVector &params, const IndexType id, const YT &val)
{
	const auto idx = static_cast<typename ParamsVector::size_type>(id);

	SetParamInternal(params, idx, val);
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::SetParamInternal(ParamsVector &params, const PVSize idx, const YT &val)
{
	assert(idx >= 0 && idx < params.size());

	if (!IsFixedParameterInternal(idx))
		params[idx] = val;
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::Reset()
{
	m_iterationCounter = 0;

	m_aborted = false;
	m_accepted = false;

	m_improvement = 0;
	m_lambda = 1. / 16384.;
	m_lambdaCoeff = 2;
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::OnParamsChanged(const bool regressCall)
{
	if (!regressCall) {
		Reset();

		m_p = YTMatrix(m_notFixed, m_x.size());
		for (PVSize i = 0, nfx = 0; i != m_params.size(); ++i) {
			if (!IsFixedParameterInternal(i))
				m_pindexes[nfx++] = i;
		}
	}

	ECHMET_TRACE_T1(RegressTracing, VECTOR_X, XT, m_x);
	ECHMET_TRACE_T1(RegressTracing, VECTOR_Y, YT, m_y);
	ECHMET_TRACE_T1(RegressTracing, VECTOR_PARAMS, ParamsVector, m_params);

	OnParamsChangedInternal();
	CalculateRSS();

	ECHMET_TRACE_T1(RegressTracing, MATRIX_P, YT, m_p);
	ECHMET_TRACE_T1(RegressTracing, VECTOR_FX, YT, m_fx);
	ECHMET_TRACE_T1(RegressTracing, VECTOR_ERROR, YT, m_error);

	if (!regressCall)
		Report();
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::OnParamsChangedInternal()
{
	CalculateFx();
	ACalculateP();
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::CalculateFx()
{
	for (size_t i = 0; i < m_x.size(); ++i)
		m_fx(i) = ACalculateFx(m_x[i], m_params, i);
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::ACalculateP()
{
	for (msize_t i = 0; i != m_notFixed; ++i) {
		const IndexType pid = static_cast<const IndexType>(m_pindexes[i]);

	msize_t xSize = static_cast<msize_t>(m_x.size());
#ifndef ECHMET_REGRESS_DISABLE_OPENMP
		#pragma omp parallel for schedule(static)
#endif // ECHMET_REGRESS_DISABLE_OPENMP
		for (msize_t k = 0; k < xSize; k++)
			m_p(i,k) = ACalculateDerivative(m_x[k], m_params, pid, k);
	}
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::CalculateRSS()
{
	m_error = m_y;
	m_error -= m_fx;

	YTMatrix RSStmp(1,1);
	RSStmp = m_error.transpose() * m_error;

	m_rss = RSStmp(0,0);
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::CheckMatrix(YTMatrix const & matrix) noexcept(false)
{
	for (int col = 0; col < matrix.cols(); col++) {
		for (int row = 0; row < matrix.rows(); row++) {
			const YT &v = matrix(row, col);

			if (std::isnan(v) || std::isinf(v))
				throw RegressException("Non-numerical values in matrix");
		}
	}
}

template <typename XT, typename YT, size_t NParams, typename IndexType>
void RegressFunction<XT, YT, NParams, IndexType>::CheckSolution(const YTMatrix &A, const YTMatrix &result, const YTMatrix &rhs) const noexcept(false)
{
	const bool solutionExists = (A * result).isApprox(rhs, 1.0e-10);

	if (!solutionExists)
		throw RegressException("No solution exists");
}

} // namespace RegressCore

} // namespace ECHMET

#endif // ECHMET_REGRESS_H
