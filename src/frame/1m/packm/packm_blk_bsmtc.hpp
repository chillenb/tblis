#ifndef _TBLIS_FRAME_1M_PACKM_BLK_BSMTC_HPP_
#define _TBLIS_FRAME_1M_PACKM_BLK_BSMTC_HPP_

#include "tblis.h"

namespace tblis
{

void packm_blk_bsmtc(const obj_t*     c,
                           obj_t*     p,
                     const cntx_t*    cntx,
                     const cntl_t*    cntl,
                           thrinfo_t* thread);

}

#endif
