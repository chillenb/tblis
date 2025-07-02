#ifndef _TBLIS_FRAME_3M_GEMM_KER_BSMTC_HPP_
#define _TBLIS_FRAME_3M_GEMM_KER_BSMTC_HPP_

#include "tblis.h"

namespace tblis
{

void gemm_ker_bsmtc
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     );

}

#endif
