#ifndef _TBLIS_INTERNAL_1T_DOT_HPP_
#define _TBLIS_INTERNAL_1T_DOT_HPP_

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"

namespace tblis
{
namespace internal
{

void dot(type_t type, const communicator& comm, const cntx_t* cntx,
         const len_vector& len_AB,
         bool conj_A, const char* A, const stride_vector& stride_A_AB,
         bool conj_B, const char* B, const stride_vector& stride_B_AB,
         char* result);

}
}

#endif
