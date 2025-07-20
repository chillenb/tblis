#ifndef _TBLIS_INTERNAL_0_MULT_HPP_
#define _TBLIS_INTERNAL_0_MULT_HPP_

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"

namespace tblis
{
namespace internal
{

void mult(type_t type, const scalar& alpha, bool conj_A, const char* A,
                                            bool conj_B, const char* B,
                       const scalar&  beta, bool conj_C,       char* C);

}
}

#endif
