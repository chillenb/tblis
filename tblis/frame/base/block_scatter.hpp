#ifndef _TBLIS_FRAME_BASE_BLOCK_SCATTER_HPP_
#define _TBLIS_FRAME_BASE_BLOCK_SCATTER_HPP_

#include "tblis.h"

namespace tblis
{

struct bsmtc_params
{
    std::array<len_type,2> nblock;
    std::array<const stride_type*,2> block_off;
    std::array<int,2> ndim;
    std::array<const len_type*,2> len;
    std::array<const stride_type*,2> stride;
    std::array<bool,2> pack_3d;
};

void fill_block_scatter(      len_type     type_size,
                              len_type     nblock,
                        const stride_type* block_off,
                              int          ndim,
                        const len_type*    len,
                        const stride_type* stride,
                              len_type     BS,
                              len_type     off,
                              len_type     size,
                              stride_type*       scat,
                              stride_type*       bs,
                              bool         pack_3d);

}

#endif
