#ifndef _TBLIS_INTERNAL_1T_DPD_SET_HPP_
#define _TBLIS_INTERNAL_1T_DPD_SET_HPP_

#include "util.hpp"

namespace tblis
{
namespace internal
{

void set(type_t type, const communicator& comm, const cntx_t* cntx,
         const scalar& alpha, const dpd_marray_view<char>& A, const dim_vector& idx_A);

}
}

#endif
