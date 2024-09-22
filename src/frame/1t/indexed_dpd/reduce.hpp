#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_REDUCE_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_REDUCE_HPP_

#include "util.hpp"

namespace tblis
{
namespace internal
{

void reduce(type_t type, const communicator& comm, const cntx_t* cntx, reduce_t op,
            const indexed_dpd_marray_view<char>& A, const dim_vector& idx_A_A,
            char* result, len_type& idx);

}
}

#endif
