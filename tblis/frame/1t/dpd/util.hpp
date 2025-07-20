#ifndef _TBLIS_INTERNAL_1T_DPD_UTIL_HPP_
#define _TBLIS_INTERNAL_1T_DPD_UTIL_HPP_

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"
#include "tblis/frame/base/tensor.hpp"

#include "marray/dpd/dpd_marray_view.hpp"

#include "tblis/frame/1t/dense/add.hpp"

namespace tblis
{

using MArray::dpd_marray_view;
using MArray::marray;
using MArray::marray_view;
using MArray::dim_vector;
using MArray::irrep_vector;
using MArray::irrep_iterator;
using MArray::matrix;

namespace internal
{

enum dpd_impl_t {BLIS, BLOCKED, FULL};
extern dpd_impl_t dpd_impl;

inline
auto block_to_full(type_t type, const communicator& comm, const cntx_t* cntx,
                   const dpd_marray_view<char>& A)
{
    auto ts = type_size[type];
    auto nirrep = A.num_irreps();
    auto ndim_A = A.dimension();

    len_vector len_A2(ndim_A);
    matrix<len_type> off_A2{ndim_A, nirrep};
    for (auto i : range(ndim_A))
    {
        for (auto irrep : range(nirrep))
        {
            off_A2[i][irrep] = len_A2[i];
            len_A2[i] += A.length(i, irrep);
        }
    }
    auto stride_A2 = MArray::detail::strides(len_A2);
    auto size_A = stl_ext::prod(len_A2);

    auto A2 = comm.master() ? new char[size_A*ts]() : nullptr;
    comm.broadcast_value(A2);

    A.for_each_block(
    [&](auto&& local_A, auto&& irreps_A)
    {
        scalar one(1.0, type);
        scalar zero(0.0, type);

        auto data_A = A.data() + (local_A.data() - A.data())*ts;
        auto data_A2 = A2;
        for (auto i : range(ndim_A))
            data_A2 += off_A2[i][irreps_A[i]]*stride_A2[i]*ts;

        add(type, comm, cntx, {}, {}, local_A.lengths(),
             one, false,  data_A, {}, local_A.strides(),
            zero, false, data_A2, {},         stride_A2);
    });

    return std::make_tuple(A2, len_A2, stride_A2);
}

inline
void full_to_block(type_t type, const communicator& comm, const cntx_t* cntx,
                   char* A2, const len_vector&, const stride_vector& stride_A2,
                   const dpd_marray_view<char>& A)
{
    auto ts = type_size[type];
    auto nirrep = A.num_irreps();
    auto ndim_A = A.dimension();

    matrix<len_type> off_A2{ndim_A, nirrep};
    for (auto i : range(ndim_A))
    {
        len_type off = 0;
        for (auto irrep : range(nirrep))
        {
            off_A2[i][irrep] = off;
            off += A.length(i, irrep);
        }
    }

    A.for_each_block(
    [&](auto&& local_A, auto&& irreps_A)
    {
        scalar one(1.0, type);
        scalar zero(0.0, type);

        auto data_A = A.data() + (local_A.data() - A.data())*ts;
        auto data_A2 = A2;
        for (auto i : range(ndim_A))
            data_A2 += off_A2[i][irreps_A[i]]*stride_A2[i]*ts;

        add(type, comm, cntx, {}, {}, local_A.lengths(),
             one, false, data_A2, {},         stride_A2,
            zero, false,  data_A, {}, local_A.strides());
    });
}

template <int I, size_t N>
void dense_total_lengths_and_strides_helper(std::array<len_vector,N>&,
                                            std::array<stride_vector,N>&) {}

template <int I, size_t N, typename Array, typename... Args>
void dense_total_lengths_and_strides_helper(std::array<len_vector,N>& len,
                                            std::array<stride_vector,N>& stride,
                                            const Array& A,
                                            const dim_vector&, const Args&... args)
{
    int ndim = A.permutation().size();
    auto nirrep = A.num_irreps();

    len[I].resize(ndim);
    stride[I].resize(ndim);

    for (auto j : range(ndim))
    {
        for (auto irrep : range(nirrep))
            len[I][j] += A.length(j, irrep);
    }

    auto iperm = MArray::detail::inverse_permutation(A.permutation());
    stride[I][iperm[0]] = 1;
    for (auto j : range(1,ndim))
    {
        stride[I][iperm[j]] = stride[I][iperm[j-1]] * len[I][iperm[j-1]];
    }

    dense_total_lengths_and_strides_helper<I+1>(len, stride, args...);
}

template <size_t N, typename... Args>
void dense_total_lengths_and_strides(std::array<len_vector,N>& len,
                                     std::array<stride_vector,N>& stride,
                                     const Args&... args)
{
    dense_total_lengths_and_strides_helper<0>(len, stride, args...);
}

template <typename T>
bool is_block_empty(const dpd_marray_view<T>& A, const irrep_vector& irreps)
{
    auto irrep = 0;

    for (auto i : range(A.dimension()))
    {
        irrep ^= irreps[i];
        if (!A.length(i, irreps[i])) return true;
    }

    return irrep != A.irrep();
}

inline int assign_irrep(int, int irrep)
{
    return irrep;
}

template <typename... Args>
int assign_irrep(int dim, int irrep,
                 irrep_vector& irreps,
                 const dim_vector& idx,
                 Args&... args)
{
    irreps[idx[dim]] = irrep;
    return assign_irrep(dim, irrep, args...);
}

template <typename... Args>
void assign_irreps(int ndim, int irrep, int nirrep,
                   stride_type block, Args&... args)
{
    int mask = nirrep-1;
    int shift = (nirrep>1) + (nirrep>2) + (nirrep>4);

    int irrep0 = irrep;
    for (auto i : range(1,ndim))
    {
        irrep0 ^= assign_irrep(i, block & mask, args...);
        block >>= shift;
    }
    if (ndim) assign_irrep(0, irrep0, args...);
}

}
}

#endif
