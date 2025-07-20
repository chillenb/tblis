#ifndef _TBLIS_INTERNAL_3T_DPD_MULT_HPP_
#define _TBLIS_INTERNAL_3T_DPD_MULT_HPP_

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"

#include "tblis/frame/1t/dpd/util.hpp"

#include "marray/dpd/dpd_marray_view.hpp"

namespace tblis
{
namespace internal
{

using MArray::dpd_marray_view;
using MArray::dim_vector;

void gemm_dpd_blis(type_t type, const communicator& comm, const cntx_t* cntx,
                   int nirrep, int irrep_AC, int irrep_BC, int irrep_AB,
                   bool pack_3d_AC, bool pack_3d_BC, bool pack_3d_AB,
                   const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
                   const dim_vector idx_A_AC, int unit_A_AC,
                   const dim_vector idx_A_AB, int unit_A_AB,
                   const dim_vector& extra_dim_A,
                   const irrep_vector& extra_irreps_A,
                   const len_vector& extra_idx_A,
                                        bool conj_B, const dpd_marray_view<char>& B,
                   const dim_vector& idx_B_BC, int unit_B_BC,
                   const dim_vector& idx_B_AB, int unit_B_AB,
                   const dim_vector& extra_dim_B,
                   const irrep_vector& extra_irreps_B,
                   const len_vector& extra_idx_B,
                                                     const dpd_marray_view<char>& C,
                   const dim_vector& idx_C_AC, int unit_C_AC,
                   const dim_vector& idx_C_BC, int unit_C_BC,
                   const dim_vector& extra_dim_C,
                   const irrep_vector& extra_irreps_C,
                   const len_vector& extra_idx_C);

void mult(type_t type, const communicator& comm, const cntx_t* cntx,
          const scalar& alpha,
          bool conj_A, const dpd_marray_view<char>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
          bool conj_B, const dpd_marray_view<char>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          const scalar&  beta,
          bool conj_C, const dpd_marray_view<char>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC);

}
}

#endif
