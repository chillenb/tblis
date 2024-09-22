#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_SHIFT_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_SHIFT_HPP_

#include "util.hpp"

namespace tblis
{
namespace internal
{

void shift(type_t type, const communicator& comm, const cntx_t* cntx,
           const scalar& alpha, const scalar& beta, bool conj_A,
           const indexed_dpd_marray_view<char>& A, const dim_vector& idx_A_A);

}
}

#endif
