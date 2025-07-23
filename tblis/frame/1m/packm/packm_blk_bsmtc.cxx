#include "packm_blk_bsmtc.hpp"

#include "tblis/plugin/bli_plugin_tblis.h"

#include "tblis/frame/base/alignment.hpp"
#include "tblis/frame/base/block_scatter.hpp"

namespace tblis
{

void packm_blk_bsmtc(const obj_t*     c,
                           obj_t*     p,
                     const cntx_t*    cntx,
                     const cntl_t*    cntl,
                           thrinfo_t* thread)
{
    auto schema = bli_packm_def_cntl_pack_schema(cntl);
    auto dt_p   = bli_packm_def_cntl_target_dt(cntl);

    auto size_p = bli_packm_init( dt_p, c, p, cntl );
    if (!size_p) return;

    auto dt_c            = bli_obj_dt( c );
         dt_p            = bli_obj_dt( p );
    auto dt_c_size       = bli_dt_size( dt_c );
    auto dt_p_size       = bli_dt_size( dt_p );

    auto conjc           = bli_obj_conj_status( c );

    auto iter_dim        = bli_obj_length( p );
    auto panel_len       = bli_obj_width( p );
    auto panel_len_max   = bli_obj_padded_width( p );
    auto panel_dim_max   = bli_obj_panel_dim( p );
    auto panel_len_block = bli_cntx_get_blksz_def_dt(dt_p, (bszid_t)KE_BSZ, cntx);

    // Compute the total number of iterations we'll need.
    auto n_iter   = ceil_div(iter_dim, panel_dim_max);
    auto k_blocks = ceil_div(panel_len, panel_len_block);

    auto panel_size = size_p;
    size_p += size_as_type<stride_type>(n_iter*(panel_dim_max+1) + k_blocks*(panel_len_block+1), dt_p) * dt_p_size;

    // Update the buffer address in p to point to the buffer associated
    // with the mem_t entry acquired from the memory broker (now cached in
    // the control tree node).
    auto buffer = bli_packm_alloc( size_p, cntl, thread );
    bli_obj_set_buffer( buffer, p );

    auto  c_cast        = static_cast<char*>(bli_obj_buffer(c));
    auto  panel_dim_off = bli_obj_row_off(c);
    auto  panel_len_off = bli_obj_col_off(c);

    auto  p_cast        = static_cast<char*>(bli_obj_buffer(p));
    auto  ldp           = bli_obj_col_stride(p);
    auto  ps_p          = bli_obj_panel_stride(p);
    auto  bcast_p       = bli_packm_def_cntl_bmult_m_bcast(cntl);

    obj_t kappa_local;
    auto  kappa_cast    = static_cast<char*>(bli_packm_scalar(&kappa_local, p));

    auto  params        = static_cast<const bsmtc_params*>(bli_packm_def_cntl_ukr_params(cntl));
    auto  packm_ker     = reinterpret_cast<packm_bsmtc_ft>(bli_cntx_get_ukr2_dt(dt_c, dt_p, PACKM_BSMTC_UKR, cntx));

    auto  rscat_c       = convert_and_align<stride_type>(p_cast + panel_size);
    auto  cscat_c       = rscat_c + n_iter*panel_dim_max;
    auto  rbs_c         = cscat_c + k_blocks*panel_len_block;
    auto  cbs_c         = rbs_c   + n_iter;

    // Query the number of threads (single-member thread teams) and the thread
    // team ids from the current thread's packm thrinfo_t node.
    const auto nt  = bli_thrinfo_num_threads(thread);
    const auto tid = bli_thrinfo_thread_id(thread);

    dim_t m_start, m_end, m_inc;
    bli_thread_range_sl(tid, nt, n_iter, 1, FALSE, &m_start, &m_end, &m_inc);

    dim_t k_start, k_end, k_inc;
    bli_thread_range_sl(tid, nt, k_blocks, 1, FALSE, &k_start, &k_end, &k_inc);

    if (m_start < n_iter)
    fill_block_scatter(dt_c_size,
                       params->nblock[0],
                       params->block_off[0],
                       params->ndim[0],
                       params->len[0],
                       params->stride[0],
                       panel_dim_max,
                       panel_dim_off + m_start*panel_dim_max,
                       std::min(m_end*panel_dim_max, iter_dim) - m_start*panel_dim_max,
                       rscat_c + m_start*panel_dim_max,
                       rbs_c + m_start,
                       params->pack_3d[0]);

    if (k_start < k_blocks)
    fill_block_scatter(dt_c_size,
                       params->nblock[1],
                       params->block_off[1],
                       params->ndim[1],
                       params->len[1],
                       params->stride[1],
                       panel_len_block,
                       panel_len_off + k_start*panel_len_block,
                       std::min(k_end*panel_len_block, panel_len) - k_start*panel_len_block,
                       cscat_c + k_start*panel_len_block,
                       cbs_c + k_start,
                       params->pack_3d[1]);

    bli_thrinfo_barrier(thread);

    // Determine the thread range and increment using the current thread's
    // packm thrinfo_t node. NOTE: The definition of bli_thread_range_slrr()
    // will depend on whether slab or round-robin partitioning was requested
    // at configure-time.
    dim_t it_start, it_end, it_inc;
    bli_thread_range_slrr(tid, nt, n_iter, 1, FALSE, &it_start, &it_end, &it_inc);

    // Iterate over every logical micropanel in the source matrix.
    for (auto it : range(n_iter))
    {
        auto panel_dim = std::min(panel_dim_max, iter_dim);

        // Hermitian/symmetric and general packing may use slab or round-
        // robin (bli_is_my_iter()), depending on which was selected at
        // configure-time.
        if (bli_is_my_iter(it, it_start, it_end, tid, nt))
        {
            packm_ker
            (
                conjc,
                schema,
                panel_dim,
                panel_len,
                panel_dim_max,
                panel_len_max,
                bcast_p,
                kappa_cast,
                c_cast, rscat_c, *rbs_c, cscat_c, cbs_c,
                p_cast, ldp
            );
        }

        p_cast   += ps_p*dt_p_size;
        rscat_c  += panel_dim_max;
        rbs_c    += 1;
        iter_dim -= panel_dim_max;
    }
}

}
