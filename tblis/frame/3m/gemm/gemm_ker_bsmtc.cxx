#include "gemm_ker_bsmtc.hpp"

#include "tblis/plugin/bli_plugin_tblis.h"

#include "tblis/frame/base/alignment.hpp"
#include "tblis/frame/base/block_scatter.hpp"

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

    auto a_cast   = static_cast<const char*>(bli_obj_buffer( a ));
    auto is_a     = bli_obj_imag_stride( a );
    auto pd_a     = bli_obj_panel_dim( a );
    auto ps_a     = bli_obj_panel_stride( a );

    auto b_cast   = static_cast<const char*>(bli_obj_buffer( b ));
    auto is_b     = bli_obj_imag_stride( b );
    auto pd_b     = bli_obj_panel_dim( b );
    auto ps_b     = bli_obj_panel_stride( b );

    auto c_cast   = static_cast<char*>(bli_obj_buffer( c ));
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

    auto gemm_ukr = reinterpret_cast<gemm_bsmtc_ft>(bli_cntx_get_ukr_dt(dt_c, GEMM_BSMTC_UKR, cntx));
    auto params   = static_cast<const bsmtc_params*>(bli_gemm_var_cntl_real_ukr(cntl) ?
                                                     bli_gemm_var_cntl_real_params(cntl) :
                                                     bli_gemm_var_cntl_params(cntl));

    //
    // Assumptions/assertions:
    //   rs_a == 1
    //   cs_a == PACKMR
    //   pd_a == MR
    //   ps_a == stride to next micro-panel of A
    //   rs_b == PACKNR
    //   cs_b == 1
    //   pd_b == NR
    //   ps_b == stride to next micro-panel of B
    //   rs_c == (no assumptions)
    //   cs_c == (no assumptions)
    //

    // Compute number of primary and leftover components of the m and n
    // dimensions.
    auto n_iter = ceil_div(n, NR);
    auto m_iter = ceil_div(m, MR);

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

    dim_t jr_start, jr_end, jr_inc;
    dim_t ir_start, ir_end, ir_inc;

#ifdef BLIS_ENABLE_JRIR_TLB

    // Query the number of threads and thread ids for the jr loop around
    // the microkernel.
    auto thread = bli_thrinfo_sub_node( 0, thread_par );
    auto jr_nt  = bli_thrinfo_n_way( thread );
    auto jr_tid = bli_thrinfo_work_id( thread );

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
    auto thread = bli_thrinfo_sub_node( 0, thread_par );
    auto caucus = bli_thrinfo_sub_node( 0, thread );
    auto jr_nt  = bli_thrinfo_n_way( thread );
    auto jr_tid = bli_thrinfo_work_id( thread );
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

    auto scat_size = sizeof(stride_type) * (m_iter*(MR+1) + n_iter*(NR+1));
    auto rscat_c = static_cast<stride_type*>(bli_packm_alloc_ex(scat_size, BLIS_BUFFER_FOR_GEN_USE, thread_par));
    auto cscat_c = rscat_c + MR*m_iter;
    auto rbs_c   = cscat_c + NR*n_iter;
    auto cbs_c   = rbs_c + m_iter;

    auto irjr_nt = ir_nt * jr_nt;
    auto irjr_tid = ir_tid + ir_nt * jr_tid;

    dim_t m_start, m_end, m_inc;
    bli_thread_range_sl(irjr_tid, irjr_nt, m_iter, 1, FALSE, &m_start, &m_end, &m_inc);

    dim_t n_start, n_end, n_inc;
    bli_thread_range_sl(irjr_tid, irjr_nt, n_iter, 1, FALSE, &n_start, &n_end, &n_inc);

    if (m_start < m_iter)
    fill_block_scatter(dt_c_size,
                       params->nblock[0],
                       params->block_off[0],
                       params->ndim[0],
                       params->len[0],
                       params->stride[0],
                       MR,
                       off_m + m_start*MR,
                       std::min(m_end*MR, m) - m_start*MR,
                       rscat_c + m_start*MR,
                       rbs_c + m_start,
                       params->pack_3d[0]);

    if (n_start < n_iter)
    fill_block_scatter(dt_c_size,
                       params->nblock[1],
                       params->block_off[1],
                       params->ndim[1],
                       params->len[1],
                       params->stride[1],
                       NR,
                       off_n + n_start*NR,
                       std::min(n_end*NR, n) - n_start*NR,
                       cscat_c + n_start*NR,
                       cbs_c + n_start,
                       params->pack_3d[1]);

    bli_thrinfo_barrier(thread_par);

    // Loop over the n dimension (NR columns at a time).
    for ( dim_t j = jr_start; j < jr_end && n_ut_for_me; j += jr_inc )
    {
        auto b1 = b_cast + j * cstep_b;

        // Compute the current microtile's width.
        auto n_cur = std::min(n - j*NR, NR);

        // Initialize our next panel of B to be the current panel of B.
        auto b2 = b1;

        // Loop over the m dimension (MR rows at a time).
        for ( dim_t i = ir_start; i < ir_end && n_ut_for_me; i += ir_inc )
        {
            auto a1  = a_cast + i * rstep_a;

            // Compute the current microtile's length.
            auto m_cur = std::min(m - i*MR, MR);

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
              c_cast, rbs_c[i], rscat_c + i*MR,
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
}

}
