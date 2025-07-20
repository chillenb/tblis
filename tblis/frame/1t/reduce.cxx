#include "reduce.h"

#include "tblis/plugin/bli_plugin_tblis.h"

#include "tblis/frame/base/tensor.hpp"

#include "tblis/frame/1t/dense/reduce.hpp"
#include "tblis/frame/1t/dpd/reduce.hpp"
#include "tblis/frame/1t/indexed/reduce.hpp"
#include "tblis/frame/1t/indexed_dpd/reduce.hpp"

namespace tblis
{

TBLIS_EXPORT
void tblis_tensor_reduce(const tblis_comm* comm,
                         const tblis_config* cntx,
                         reduce_t op,
                         const tblis_tensor* A,
                         const label_type* idx_A_,
                         tblis_scalar* result,
                         len_type* idx)
{
    internal::initialize_once();

    TBLIS_ASSERT(A->type == result->type);

    auto ndim_A = A->ndim;
    len_vector len_A;
    stride_vector stride_A;
    label_vector idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    if (idx_A.empty())
    {
        len_A.push_back(1);
        stride_A.push_back(0);
        idx_A.push_back(0);
    }

    fold(len_A, idx_A, stride_A);

    if (A->scalar.is_negative())
    {
        if (op == REDUCE_MIN) op = REDUCE_MAX;
        else if (op == REDUCE_MAX) op = REDUCE_MIN;
    }

    parallelize_if(
    [&](const communicator& comm)
    {
        internal::reduce(A->type, comm, bli_gks_query_cntx(), op, len_A,
                         reinterpret_cast<char*>(A->data), stride_A,
                         result->raw(), *idx);
    }, comm);

    if (A->conj) result->conj();

    if (op == REDUCE_SUM_ABS || op == REDUCE_MAX_ABS || op == REDUCE_MIN_ABS || op == REDUCE_NORM_2)
        *result *= abs(A->scalar);
    else
        *result *= A->scalar;
}

template <typename T>
void reduce(const communicator& comm, reduce_t op,
            dpd_marray_view<const T> A, const label_vector& idx_A,
            T& result, len_type& idx)
{
    internal::initialize_once();

    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::reduce(type_tag<T>::value, comm, bli_gks_query_cntx(), op,
                     reinterpret_cast<dpd_marray_view<char>&>(A), idx_A_A,
                     reinterpret_cast<char*>(&result), idx);
}

#undef FOREACH_TYPE
#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, reduce_t op, \
                     dpd_marray_view<const T> A, const label_vector& idx_A, \
                     T& result, len_type& idx);
DO_FOREACH_TYPE

template <typename T>
void reduce(const communicator& comm, reduce_t op,
            indexed_marray_view<const T> A, const label_vector& idx_A,
            T& result, len_type& idx)
{
    internal::initialize_once();

    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::reduce(type_tag<T>::value, comm, bli_gks_query_cntx(), op,
                     reinterpret_cast<indexed_marray_view<char>&>(A), idx_A_A,
                     reinterpret_cast<char*>(&result), idx);
}

#undef FOREACH_TYPE
#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, reduce_t op, \
                     indexed_marray_view<const T> A, const label_vector& idx_A, \
                     T& result, len_type& idx);
DO_FOREACH_TYPE

template <typename T>
void reduce(const communicator& comm, reduce_t op,
            indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
            T& result, len_type& idx)
{
    internal::initialize_once();

    (void)idx_A;

    auto ndim_A = A.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    internal::reduce(type_tag<T>::value, comm, bli_gks_query_cntx(), op,
                     reinterpret_cast<indexed_dpd_marray_view<char>&>(A), idx_A_A,
                     reinterpret_cast<char*>(&result), idx);
}

#undef FOREACH_TYPE
#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, reduce_t op, \
                     indexed_dpd_marray_view<const T> A, const label_vector& idx_A, \
                     T& result, len_type& idx);
DO_FOREACH_TYPE

}
