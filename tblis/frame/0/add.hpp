#ifndef _TBLIS_INTERNAL_0_ADD_HPP_
#define _TBLIS_INTERNAL_0_ADD_HPP_

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"

namespace tblis
{
namespace internal
{

void add(type_t type, const scalar& alpha, bool conj_A, const char* A,
                      const scalar&  beta, bool conj_B,       char* B);

}
}

#endif
