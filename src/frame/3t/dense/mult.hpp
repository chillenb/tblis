#ifndef _TBLIS_INTERNAL_3T_MULT_HPP_
#define _TBLIS_INTERNAL_3T_MULT_HPP_

#include "frame/base/thread.h"
#include "frame/base/basic_types.h"

namespace tblis
{
namespace internal
{

enum impl_t {BLIS_BASED, BLAS_BASED, REFERENCE};
extern impl_t impl;

void gemm_bsmtc_blis(type_t type, const communicator& comm, const cntx_t* cntx,
                     len_type nblock_AC, int ndim_AC, const len_type* len_AC, bool pack_3d_AC,
                     len_type nblock_BC, int ndim_BC, const len_type* len_BC, bool pack_3d_BC,
                     len_type nblock_AB, int ndim_AB, const len_type* len_AB, bool pack_3d_AB,
                     const scalar& alpha, bool conj_A, const char* A, const stride_type* block_off_A_AC, const stride_type* block_off_A_AB, const stride_type* stride_A_AC, const stride_type* stride_A_AB,
                                          bool conj_B, const char* B, const stride_type* block_off_B_BC, const stride_type* block_off_B_AB, const stride_type* stride_B_BC, const stride_type* stride_B_AB,
                     const scalar&  beta, bool conj_C,       char* C, const stride_type* block_off_C_AC, const stride_type* block_off_C_BC, const stride_type* stride_C_AC, const stride_type* stride_C_BC);

void mult(type_t type, const communicator& comm, const cntx_t* cntx,
          const len_vector& len_AB,
          const len_vector& len_AC,
          const len_vector& len_BC,
          const len_vector& len_ABC,
          const scalar& alpha, bool conj_A, const char* A,
          const stride_vector& stride_A_AB,
          const stride_vector& stride_A_AC,
          const stride_vector& stride_A_ABC,
                               bool conj_B, const char* B,
          const stride_vector& stride_B_AB,
          const stride_vector& stride_B_BC,
          const stride_vector& stride_B_ABC,
          const scalar&  beta, bool conj_C,       char* C,
          const stride_vector& stride_C_AC,
          const stride_vector& stride_C_BC,
          const stride_vector& stride_C_ABC);

}
}

#endif
