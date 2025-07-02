#ifndef _TBLIS_INTERNAL_1T_INDEXED_REDUCE_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_REDUCE_HPP_

#include "util.hpp"

namespace tblis
{
namespace internal
{

void reduce(type_t type, const communicator& comm, const cntx_t* cntx, reduce_t op,
            const indexed_marray_view<char>& A, const dim_vector&,
            char* result, len_type& idx);

}
}

#endif
