#ifndef _TBLIS_INTERNAL_0_REDUCE_HPP_
#define _TBLIS_INTERNAL_0_REDUCE_HPP_

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"

namespace tblis
{
namespace internal
{

void reduce(type_t type, reduce_t op,
            const char* A, len_type  idx_A,
                  char* B, len_type& idx_B);

}
}

#endif
