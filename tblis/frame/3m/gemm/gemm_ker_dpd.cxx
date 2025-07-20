#include "gemm_ker_dpd.hpp"

#include "tblis/plugin/bli_plugin_tblis.h"

#include "tblis/frame/base/alignment.hpp"
#include "tblis/frame/base/dpd_block_scatter.hpp"

namespace tblis
{

void gemm_ker_dpd
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{
    auto dt_a     = bli_obj_dt( a );
    auto dt_b     = bli_obj_dt( b );
    auto dt_c     = bli_obj_dt( c );

    auto schema_a = bli_obj_pack_schema( a );
    auto schema_b = bli_obj_pack_schema( b );

    auto m        = bli_obj_length( c );
    auto n        = bli_obj_width( c );
    auto k        = bli_obj_width( a );

    auto a_cast   = static_cast<const char*>(bli_obj_buffer_at_off( a ));
    auto is_a     = bli_obj_imag_stride( a );
    auto pd_a     = bli_obj_panel_dim( a );
    auto ps_a     = bli_obj_panel_stride( a );

    auto b_cast   = static_cast<const char*>(bli_obj_buffer_at_off( b ));
    auto is_b     = bli_obj_imag_stride( b );
    auto pd_b     = bli_obj_panel_dim( b );
    auto ps_b     = bli_obj_panel_stride( b );

    auto off_m    = bli_obj_row_off(c);
    auto off_n    = bli_obj_col_off(c);

    // Detach and multiply the scalars attached to A and B.
    // NOTE: We know that the internal scalars of A and B are already of the
    // target datatypes because the necessary typecasting would have already
    // taken place during bli_packm_init().
    obj_t scalar_a, scalar_b;
    bli_obj_scalar_detach( a, &scalar_a );
    bli_obj_scalar_detach( b, &scalar_b );
    bli_mulsc( &scalar_a, &scalar_b );

    // Grab the addresses of the internal scalar buffers for the scalar
    // merged above and the scalar attached to C.
    // NOTE: We know that scalar_b is of type dt_comp due to the above code
    // that casts the scalars of A and B to dt_comp via scalar_a and scalar_b,
    // and we know that the internal scalar in C is already of the type dt_c
    // due to the casting in the implementation of bli_obj_scalar_attach().
    auto alpha_cast = static_cast<const char*>(bli_obj_internal_scalar_buffer( &scalar_b ));
    auto beta_cast  = static_cast<const char*>(bli_obj_internal_scalar_buffer( c ));

    auto dt_a_size = bli_dt_size( dt_a );
    auto dt_b_size = bli_dt_size( dt_b );
    auto dt_c_size = bli_dt_size( dt_c );

    // Alias some constants to simpler names.
    auto MR = pd_a;
    auto NR = pd_b;

    auto  gemm_ukr = reinterpret_cast<gemm_bsmtc_ft>(bli_cntx_get_ukr_dt(dt_c, GEMM_BSMTC_UKR, cntx));
    auto& params   = *static_cast<const dpd_params*>(bli_gemm_var_cntl_real_ukr(cntl) ?
                                                     bli_gemm_var_cntl_real_params(cntl) :
                                                     bli_gemm_var_cntl_params(cntl));

    // Determine some increments used to step through A, B, and C.
    auto rstep_a = ps_a * dt_a_size;
    auto cstep_b = ps_b * dt_b_size;

    auxinfo_t aux;

    // Save the pack schemas of A and B to the auxinfo_t object.
    bli_auxinfo_set_schema_a( schema_a, &aux );
    bli_auxinfo_set_schema_b( schema_b, &aux );

    // Save the imaginary stride of A and B to the auxinfo_t object.
    bli_auxinfo_set_is_a( is_a, &aux );
    bli_auxinfo_set_is_b( is_b, &aux );

    // Save the virtual microkernel address and the params.
    bli_auxinfo_set_params( cntl, &aux );

    // Query the number of threads and thread ids for the jr loop around
    // the microkernel.
    auto thread = bli_thrinfo_sub_node( 0, thread_par );
    auto jr_nt  = bli_thrinfo_n_way( thread );
    auto jr_tid = bli_thrinfo_work_id( thread );

    len_type m_iter_all, m_patch0, m_patch_off0;
    std::tie(m_iter_all,
             m_patch0,
             m_patch_off0) = get_patches(m, off_m, MR, params.patch_size[0]);
    len_type n_iter_all, n_patch0, n_patch_off0;
    std::tie(n_iter_all,
             n_patch0,
             n_patch_off0) = get_patches(n, off_n, NR, params.patch_size[1]);

    auto scat_size = sizeof(stride_type) * (m + n + m_iter_all + n_iter_all);
    auto rscat_c = static_cast<stride_type*>(bli_packm_alloc_ex(scat_size, BLIS_BUFFER_FOR_GEN_USE, thread_par));
    auto cscat_c = rscat_c + m;
    auto rbs_c   = cscat_c + n;
    auto cbs_c   = rbs_c + m_iter_all;

    auto b0 = b_cast;
    for_each_patch(n, params.patch_size[1], n_patch0, n_patch_off0,
    [&](auto n_patch, auto n_patch_size, auto n_patch_off)
    {
        auto n_iter = ceil_div(n_patch_size, NR);

        auto a0 = a_cast;
        for_each_patch(m, params.patch_size[0], m_patch0, m_patch_off0,
        [&](int m_patch, len_type m_patch_size, len_type m_patch_off)
        {
            auto m_iter = ceil_div(m_patch_size, MR);

            dim_t jr_start, jr_end, jr_inc;
            dim_t ir_start, ir_end, ir_inc;

#ifdef BLIS_ENABLE_JRIR_TLB

            auto ir_nt  = 1;
            auto ir_tid = 0;

            auto n_ut_for_me
            =
            bli_thread_range_tlb_d( jr_nt, jr_tid, m_iter, n_iter, MR, NR,
                                    &jr_start, &ir_start );

            // Always increment by 1 in both dimensions.
            jr_inc = 1;
            ir_inc = 1;

            // Each thread iterates over the entire panel of C until it exhausts its
            // assigned set of microtiles.
            jr_end = n_iter;
            ir_end = m_iter;

            // Successive iterations of the ir loop should start at 0.
            auto ir_next = 0;

#else // ifdef ( _SLAB || _RR )

            // Query the number of threads and thread ids for the ir loop around
            // the microkernel.
            auto caucus = bli_thrinfo_sub_node( 0, thread );
            auto ir_nt  = bli_thrinfo_n_way( caucus );
            auto ir_tid = bli_thrinfo_work_id( caucus );

            // Determine the thread range and increment for the 2nd and 1st loops.
            // NOTE: The definition of bli_thread_range_slrr() will depend on whether
            // slab or round-robin partitioning was requested at configure-time.
            bli_thread_range_slrr( jr_tid, jr_nt, n_iter, 1, FALSE, &jr_start, &jr_end, &jr_inc );
            bli_thread_range_slrr( ir_tid, ir_nt, m_iter, 1, FALSE, &ir_start, &ir_end, &ir_inc );

            // Calculate the total number of microtiles assigned to this thread.
            auto n_ut_for_me = ( ( ir_end + ir_inc - 1 - ir_start ) / ir_inc ) *
                               ( ( jr_end + jr_inc - 1 - jr_start ) / jr_inc );

            // Each succesive iteration of the ir loop always starts at ir_start.
            auto ir_next = ir_start;

#endif

            bli_thrinfo_barrier(thread_par);

            auto c0 = fill_block_scatter(dt_c_size, bli_thrinfo_am_chief(thread), params, MR, NR,
                                         m_patch, m_patch_off, m_patch_size,
                                         n_patch, n_patch_off, n_patch_size,
                                         rscat_c, cscat_c, rbs_c, cbs_c);

            bli_thrinfo_barrier(thread_par);

            // Loop over the n dimension (NR columns at a time).
            for ( dim_t j = jr_start; j < jr_end && n_ut_for_me; j += jr_inc )
            {
                auto b1 = b0 + j * cstep_b;

                // Compute the current microtile's width.
                auto n_cur = std::min(n_patch_size - j*NR, NR);

                // Initialize our next panel of B to be the current panel of B.
                auto b2 = b1;

                // Loop over the m dimension (MR rows at a time).
                for ( dim_t i = ir_start; i < ir_end && n_ut_for_me; i += ir_inc )
                {
                    auto a1  = a0 + i * rstep_a;

                    // Compute the current microtile's length.
                    auto m_cur = std::min(m_patch_size - i*MR, MR);

                    // Compute the addresses of the next panels of A and B.
                    auto a2 = bli_gemm_get_next_a_upanel( a1, rstep_a, ir_inc );
                    if ( bli_is_last_iter_slrr( i, ir_end, ir_tid, ir_nt ) )
                    {
                        a2 = a_cast;
                        b2 = bli_gemm_get_next_b_upanel( b1, cstep_b, jr_inc );
                    }

                    // Save addresses of next panels of A and B to the auxinfo_t
                    // object.
                    bli_auxinfo_set_next_a( a2, &aux );
                    bli_auxinfo_set_next_b( b2, &aux );

                    // Edge case handling now occurs within the microkernel itself.
                    // Invoke the gemm micro-kernel.
                    gemm_ukr
                    (
                      m_cur,
                      n_cur,
                      k,
                      alpha_cast,
                      a1,
                      b1,
                      beta_cast,
                      c0, rbs_c[i], rscat_c + i*MR,
                          cbs_c[j], cscat_c + j*NR,
                      &aux,
                      cntx
                    );

                    // Decrement the number of microtiles assigned to the thread; once
                    // it reaches zero, return immediately.
                    n_ut_for_me--;
                }

                ir_start = ir_next;
            }

            a0 += m_iter * rstep_a;
        });

        b0 += n_iter * cstep_b;
    });
}

}
