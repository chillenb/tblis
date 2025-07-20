#include "block_scatter.hpp"
#include "alignment.hpp"

#include "tblis/plugin/bli_plugin_tblis.h"

#include <numeric>

namespace tblis
{

static void fill_scatter_3d(      len_type     type_size,
                                  int          ndim1,
                                  len_type     m0,
                            const len_type*    len,
                                  len_type     s0,
                            const stride_type* stride,
                                  len_type     BS,
                                  len_type     off,
                                  len_type     size,
                                  len_type     scat0,
                                  stride_type*       scat)
{
    const auto CL = 64/type_size;

    TBLIS_ASSERT(ndim1 > 0);

    auto m1 = *len++;
    auto s1 = *stride++;

    auto BS0 = std::min(lcm(BS, CL), std::max(BS, m0 - m0%BS));
    auto BS1 = std::min(        CL , std::max(BS, m1 - m1%BS));

    viterator<> it(len_vector{len, len+ndim1-1}, stride_vector{stride, stride+ndim1-1});

    auto p01 = off%(m0*m1);
    auto p0 = p01%m0;
    auto p1 = p01/m0;
    auto q01 = (off+size-1)%(m0*m1);
    auto q0 = q01%m0;
    auto q1 = q01/m0;
    auto r0 = q0 - q0%BS0;
    auto pos = scat0;
    auto b_min = off/(m0*m1);
    auto b_max = (off+size-1)/(m0*m1)+1;

    if (p0 == 0) p1--;
    if (q0 == m0-1) q1++;

    it.position(b_min, pos);
    auto it0 = it;
    auto pos0 = pos;

    /*
     *                  p0  q0      r0
     *    +---+---+---+---+---+---+---+
     *    |   |   |   |   |   |   |   |
     *    +---+---+---#===============#
     * p1 |   |   |   #p01| C | C | C #
     *    #=======#===#===#=======#===#
     *    # A | A # A | A # A | A # B #
     *    #---+---#---+---#---+---#---#
     *    # A | A # A | A # A | A # B #
     *    #=======#=======#=======#===#
     *    # A'| A'# A'| A'# A'| A'# B'#
     *    #=======#=======#===#===#===#
     * q1 # D | D | D | D |q01#   |   |
     *    #===================#---+---+
     *    |   |   |   |   |   |   |   |
     *    +---+---+---+---+---+---+---+
     */

    /*
     * A:  Full BS0*BS1 blocks
     * A': Partial BS0*n blocks
     */

    auto idx = 0;

    if (r0 > 0)
    {
        for (auto b = b_min;b < b_max && it.next(pos);b++)
        {
            auto min1 = b == b_min ? p1+1 : 0;
            auto max1 = b == b_max-1 ? q1 : m1;

            for (auto i1 = min1;i1 < max1;i1 += BS1)
            for (auto i0 = 0;i0 < r0;i0 += BS0)
            for (auto j1 = 0;j1 < std::min(BS1, max1-i1);j1++)
            for (auto j0 = 0;j0 < BS0;j0++)
                scat[idx++] = pos + (i0+j0)*s0 + (i1+j1)*s1;
        }

        it = it0;
        pos = pos0;
    }

    /*
     * B:  Partial m*BS1 blocks
     * B': Partial m*n block
     * C:  First partial row
     * D:  Last partial row
     */

    for (len_type b = b_min;b < b_max && it.next(pos);b++)
    {
        len_type min1 = b == b_min ? p1+1 : 0;
        len_type max1 = b == b_max-1 ? q1 : m1;

        for (len_type j1 = min1;j1 < max1;j1++)
        for (len_type j0 = r0;j0 < m0;j0++)
            scat[idx++] = pos + j0*s0 + j1*s1;

        bool do_p = b == b_min && p0 > 0;
        bool do_q = b == b_max-1 && q0 < m0-1;

        if (do_p && do_q && p1 == q1)
        {
            // p01 and q01 are on the same row
            for (len_type j0 = p0;j0 <= q0;j0++)
                scat[idx++] = pos + j0*s0 + p1*s1;
        }
        else
        {
            if (do_p)
                for (len_type j0 = p0;j0 < m0;j0++)
                    scat[idx++] = pos + j0*s0 + p1*s1;

            if (do_q)
                for (len_type j0 = 0;j0 <= q0;j0++)
                    scat[idx++] = pos + j0*s0 + q1*s1;
        }
    }

    TBLIS_ASSERT(idx == size);
}

static void fill_scatter(      len_type     type_size,
                               len_type     nblock,
                         const stride_type* block_off,
                               int          ndim,
                         const len_type*    len,
                         const stride_type* stride,
                               len_type     BS,
                               len_type     off,
                               len_type     size,
                               stride_type*       scat,
                               bool         pack_3d)
{
    TBLIS_ASSERT(off >= 0);
    TBLIS_ASSERT(size >= 0);

    if (size == 0) return;

    if (ndim == 0)
    {
        TBLIS_ASSERT(off+size <= nblock);

        for (auto b : MArray::range(size))
            scat[b] = block_off[off+b];
        return;
    }

    auto tot_len = std::reduce(len, len+ndim, len_type{1}, std::multiplies<len_type>{});
    TBLIS_ASSERT(off+size <= nblock*tot_len);

    auto m0 = *len++;
    auto s0 = *stride++;

    len_type b0, off_b;
    MArray::detail::divide<len_type>(off, tot_len, b0, off_b);

    for (auto b = b0;size > 0;b++)
    {
        auto size_b = std::min(size, tot_len-off_b);
        auto scat0 = block_off[b];

        if (pack_3d)
        {
            fill_scatter_3d(type_size,
                            ndim-1,
                            m0,
                            len,
                            s0,
                            stride,
                            BS,
                            off_b,
                            size_b,
                            scat0,
                            scat);
        }
        else
        {
            viterator<> it(len_vector{len, len+ndim-1}, stride_vector{stride, stride+ndim-1});

            len_type off0, p0;
            MArray::detail::divide<len_type>(off_b, m0, off0, p0);
            auto pos = scat0;
            it.position(off0, pos);

            for (len_type idx = 0;idx < size_b && it.next(pos);)
            {
                auto pos2 = pos + p0*s0;
                auto imax = std::min(m0-p0, size_b-idx);
                for (len_type i = 0;i < imax;i++)
                {
                    scat[idx++] = pos2;
                    pos2 += s0;
                }
                p0 = 0;
            }
        }

        scat += size_b;
        size -= size_b;
        off_b = 0;
    }
}

static void fill_block_stride(len_type BS,
                              len_type size,
                              stride_type*   scat,
                              stride_type*   bs)
{
    if (size == 0) return;

    for (len_type i = 0, ib = 0;i < size;i += BS, ib++)
    {
        auto bl = std::min(size-i, BS);
        auto s = bl > 1 ? scat[i+1]-scat[i] : 1;
        for (len_type j = i+1;j+1 < i+bl;j++)
        {
            if (scat[j+1]-scat[j] != s) s = 0;
        }
        bs[ib] = s;
    }
}

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
                              bool         pack_3d)
{
    if (size == 0) return;

    fill_scatter(type_size, nblock, block_off, ndim, len, stride, BS, off, size, scat, pack_3d);
    fill_block_stride(BS, size, scat, bs);
}

}
