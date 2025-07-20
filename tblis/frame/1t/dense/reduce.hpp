#ifndef _TBLIS_INTERNAL_1T_REDUCE_HPP_
#define _TBLIS_INTERNAL_1T_REDUCE_HPP_

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"

namespace tblis
{
namespace internal
{

void reduce(type_t type, const communicator& comm, const cntx_t* cntx, reduce_t op,
            const len_vector& len_A,
            const char* A, const stride_vector& stride_A,
            char* result, len_type& idx);

}
}

#endif
