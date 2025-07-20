#ifndef _TBLIS_INTERNAL_1T_SCALE_HPP_
#define _TBLIS_INTERNAL_1T_SCALE_HPP_

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"

namespace tblis
{
namespace internal
{

void scale(type_t type, const communicator& comm, const cntx_t* cntx,
           const len_vector& len_A, const scalar& alpha,
           bool conj_A, char* A, const stride_vector& stride_A);

}
}

#endif
