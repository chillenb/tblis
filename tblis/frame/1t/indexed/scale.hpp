#ifndef _TBLIS_INTERNAL_1T_INDEXED_SCALE_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_SCALE_HPP_

#include "util.hpp"

namespace tblis
{
namespace internal
{

void scale(type_t type, const communicator& comm, const cntx_t* cntx,
           const scalar& alpha, bool conj_A, const indexed_marray_view<char>& A,
           const dim_vector&);

}
}

#endif
