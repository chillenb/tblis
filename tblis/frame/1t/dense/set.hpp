#ifndef _TBLIS_INTERNAL_1T_SET_HPP_
#define _TBLIS_INTERNAL_1T_SET_HPP_

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"

namespace tblis
{
namespace internal
{

void set(type_t type, const communicator& comm, const cntx_t* cntx,
         const len_vector& len_A,
         const scalar& alpha, char* A, const stride_vector& stride_A);

}
}

#endif
