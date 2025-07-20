#ifndef _TBLIS_INTERNAL_1T_ADD_HPP_
#define _TBLIS_INTERNAL_1T_ADD_HPP_

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"

namespace tblis
{
namespace internal
{

void add(type_t type, const communicator& comm, const cntx_t* cntx,
         const len_vector& len_A,
         const len_vector& len_B,
         const len_vector& len_AB,
         const scalar& alpha, bool conj_A, const char* A,
         const stride_vector& stride_A,
         const stride_vector& stride_A_AB,
         const scalar&  beta, bool conj_B,       char* B,
         const stride_vector& stride_B,
         const stride_vector& stride_B_AB);

}
}

#endif
