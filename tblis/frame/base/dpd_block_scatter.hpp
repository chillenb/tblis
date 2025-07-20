#ifndef _TBLIS_FRAME_BASE_DPD_BLOCK_SCATTER_HPP_
#define _TBLIS_FRAME_BASE_DPD_BLOCK_SCATTER_HPP_

#include "tblis.h"

#include "alignment.hpp"

#include "tblis/frame/1t/dpd/util.hpp"

namespace tblis
{

struct dpd_params
{
    const dpd_marray_view<char>* tensor;
    std::array<dim_vector, 2> dims;
    dim_vector extra_dims;
    irrep_vector extra_irreps;
    len_vector extra_idx;
    std::array<int, 2> irrep;
    std::array<len_vector, 2> patch_size;
    std::array<len_vector, 2> patch_idx;
    std::array<bool, 2> pack_3d;

    dpd_params(const dpd_marray_view<char>& other,
               const dim_vector& row_inds,
               const dim_vector& col_inds,
               int col_irrep,
               const dim_vector& extra_inds,
               const irrep_vector& extra_irreps,
               const len_vector& extra_idx,
               bool pack_m_3d, bool pack_n_3d);

    void swap(dpd_params& other);

    void transpose();
};

template <typename Func>
void for_each_patch(len_type len, const len_vector& patch_size, len_type patch, len_type off, Func&& func)
{
    auto left = len;
    while (left > 0)
    {
        auto size = std::min(patch_size[patch]-off, left);

        func(patch, size, off);

        off = 0;
        left -= size;
        patch++;
    }
}

inline auto get_patches(len_type len, len_type off, len_type MR, const len_vector& patch_size)
{
    len_type idx = 0;
    while (off >= patch_size[idx]) off -= patch_size[idx++];

    len_type niter = 0;
    for_each_patch(len, patch_size, idx, off,
    [&](auto, auto size, auto)
    {
        niter += ceil_div(size, MR);
    });

    return std::make_tuple(niter, idx, off);
}

char* fill_block_scatter(len_type type_size, bool fill, const dpd_params& params,
                         len_type MR, len_type NR,
                         len_type m_patch, len_type m_off_patch, len_type m_patch_size,
                         len_type n_patch, len_type n_off_patch, len_type n_patch_size,
                         stride_type* rscat, stride_type* cscat,
                         stride_type* rbs, stride_type* cbs);

}

#endif
