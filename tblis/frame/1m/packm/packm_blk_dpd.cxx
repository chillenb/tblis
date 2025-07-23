#include "packm_blk_dpd.hpp"

#include "tblis/plugin/bli_plugin_tblis.h"

#include "tblis/frame/base/alignment.hpp"
#include "tblis/frame/base/dpd_block_scatter.hpp"

namespace tblis
{

void packm_blk_dpd(const obj_t*     c,
                         obj_t*     p,
                   const cntx_t*    cntx,
                   const cntl_t*    cntl,
                         thrinfo_t* thread)
{
    auto  schema          = bli_packm_def_cntl_pack_schema(cntl);
    auto  dt_p            = bli_packm_def_cntl_target_dt(cntl);

    auto  size_p          = bli_packm_init( dt_p, c, p, cntl );
    if (!size_p) return;

    auto  dt_c            = bli_obj_dt( c );
          dt_p            = bli_obj_dt( p );
    auto  dt_c_size       = bli_dt_size( dt_c );
    auto  dt_p_size       = bli_dt_size( dt_p );

    auto  conjc           = bli_obj_conj_status( c );

    auto  iter_dim        = bli_obj_length( p );
    auto  panel_len       = bli_obj_width( p );
    auto  panel_dim_max   = bli_obj_panel_dim( p );
	auto  panel_dim_pack  = bli_packm_def_cntl_bmult_m_pack( cntl );
    auto  panel_len_block = bli_cntx_get_blksz_def_dt(dt_p, (bszid_t)KE_BSZ, cntx);

    auto  panel_dim_off   = bli_obj_row_off(c);
    auto  panel_len_off   = bli_obj_col_off(c);

    auto  ldp             = bli_obj_col_stride(p);
    auto  ps_p            = bli_obj_panel_stride(p);
    auto  bcast_p         = bli_packm_def_cntl_bmult_m_bcast(cntl);

    obj_t kappa_local;
    auto  kappa_cast      = static_cast<char*>(bli_packm_scalar(&kappa_local, p));

    auto& params          = *static_cast<const dpd_params*>(bli_packm_def_cntl_ukr_params(cntl));
    auto  packm_ker       = reinterpret_cast<packm_bsmtc_ft>(bli_cntx_get_ukr2_dt(dt_c, dt_p, PACKM_BSMTC_UKR, cntx));

    // Compute the total number of iterations we'll need.
    len_type n_iter, n_patch0, n_patch_off0;
    std::tie(n_iter,
             n_patch0,
             n_patch_off0) = get_patches( iter_dim, panel_dim_off,   panel_dim_max, params.patch_size[0]);
    len_type k_blocks, k_patch0, k_patch_off0;
    std::tie(k_blocks,
             k_patch0,
             k_patch_off0) = get_patches(panel_len, panel_len_off, panel_len_block, params.patch_size[1]);

    auto panel_size = n_iter * ps_p * dt_p_size;
    size_p += size_as_type<stride_type>(iter_dim + panel_len + n_iter + k_blocks, dt_p) * dt_p_size;

    // Update the buffer address in p to point to the buffer associated
    // with the mem_t entry acquired from the memory broker (now cached in
    // the control tree node).
    auto buffer = bli_packm_alloc( size_p, cntl, thread );
    bli_obj_set_buffer( buffer, p );

    auto p_cast  = static_cast<char*>(bli_obj_buffer(p));
    auto rscat_c = convert_and_align<stride_type>(p_cast + panel_size);
    auto cscat_c = rscat_c + iter_dim;
    auto rbs_c   = cscat_c + panel_len;
    auto cbs_c   = rbs_c   + n_iter;

    // Query the number of threads (single-member thread teams) and the thread
    // team ids from the current thread's packm thrinfo_t node.
    const auto nt  = bli_thrinfo_num_threads(thread);
    const auto tid = bli_thrinfo_thread_id(thread);

    auto p0 = p_cast;

    for_each_patch(iter_dim, params.patch_size[0], n_patch0, n_patch_off0,
    [&](auto n_patch, auto n_patch_size, auto n_patch_off)
    {
        char* p1 = p0;

        for_each_patch(panel_len, params.patch_size[1], k_patch0, k_patch_off0,
        [&](int k_patch, len_type k_patch_size, len_type k_patch_off)
        {
            TBLIS_ASSERT(p1 >= p_cast);
            TBLIS_ASSERT(p1 + k_patch_size * panel_dim_pack * dt_p_size <=
                         p0 + panel_size * dt_p_size);

            bli_thrinfo_barrier(thread);

            auto p2 = p1;
            auto c2 = fill_block_scatter(dt_c_size, bli_thrinfo_am_chief(thread), params,
                                         panel_dim_max, panel_len_block,
                                         n_patch, n_patch_off, n_patch_size,
                                         k_patch, k_patch_off, k_patch_size,
                                         rscat_c, cscat_c, rbs_c, cbs_c);
            auto rscat_c2 = rscat_c;
            auto rbs_c2   = rbs_c;

            bli_thrinfo_barrier(thread);

            // Determine the thread range and increment using the current thread's
            // packm thrinfo_t node. NOTE: The definition of bli_thread_range_slrr()
            // will depend on whether slab or round-robin partitioning was requested
            // at configure-time.
            dim_t it_start, it_end, it_inc;
            bli_thread_range_slrr(tid, nt, n_iter, 1, FALSE, &it_start, &it_end, &it_inc);

            // Iterate over every logical micropanel in the source matrix.
            auto left = n_patch_size;
            for (auto it = 0; left > 0; it++)
            {
                auto panel_dim = std::min<len_type>(panel_dim_max, left);

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
                        k_patch_size,
                        panel_dim_max,
                        k_patch_size,
                        bcast_p,
                        kappa_cast,
                        c2, rscat_c2, *rbs_c2, cscat_c, cbs_c,
                        p2, ldp
                    );
                }

                p2       += ps_p * dt_p_size;
                rscat_c2 += panel_dim_max;
                rbs_c2   += 1;
                left     -= panel_dim;
            }

            p1 += k_patch_size * panel_dim_pack * dt_p_size;
        });

        p0 += ceil_div(n_patch_size, panel_dim_max) * ps_p * dt_p_size;
    });
}

}
