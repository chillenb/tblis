#ifndef _TBLIS_PLUGIN_PLUGIN_HPP_
#define _TBLIS_PLUGIN_PLUGIN_HPP_

#ifdef __clang__
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

#include "tblis.h"

#define BLIS_ENABLE_STD_COMPLEX
#include "blis.h"

namespace tblis
{

//
// Kernel and blocksize IDs
//

extern kerid_t GEMM_BSMTC_UKR;
extern kerid_t PACKM_BSMTC_UKR;
extern kerid_t MULT_KER;
extern kerid_t REDUCE_KER;
extern kerid_t SHIFT_KER;
extern kerid_t TRANS_KER;
extern kerid_t MRT_BSZ;
extern kerid_t NRT_BSZ;
extern kerid_t KE_BSZ;

//
// Kernels and structures
//

struct bsmtc_auxinfo_t : auxinfo_t
{
    const stride_type* rscat;
    const stride_type* cscat;
    const stride_type* rbs;
    const stride_type* cbs;
};

using gemm_bsmtc_ft = void(*)
    (
            dim_t      m,
            dim_t      n,
            dim_t      k,
      const void*      alpha,
      const void*      a,
      const void*      b,
      const void*      beta0,
            void*      c0, stride_type rs_c, const stride_type* rscat_c,
                           stride_type cs_c, const stride_type* cscat_c,
      const auxinfo_t* auxinfo,
      const cntx_t*    cntx
    );

using packm_bsmtc_ft = void(*)
    (
            conj_t conjc,
            pack_t schema,
            dim_t  panel_dim,
            dim_t  panel_len,
            dim_t  panel_dim_max,
            dim_t  panel_len_max,
            dim_t  panel_bcast,
      const void*  kappa_,
      const void*  c_, const stride_type* rscat_c,       stride_type  rbs_c,
                       const stride_type* cscat_c, const stride_type* cbs_c,
            void*  p_,       stride_type  ldp
    );

using mult_ft = void(*)
    (
            len_type n,
      const void*    alpha_, bool conj_A, const void* A_, stride_type inc_A,
                             bool conj_B, const void* B_, stride_type inc_B,
      const void*     beta_, bool conj_C,       void* C_, stride_type inc_C
    );

using reduce_ft = void(*)
    (
            reduce_t  op,
            len_type  n,
      const void*     A_, stride_type inc_A,
            void*     value_,
            len_type& idx_
    );

using shift_ft = void(*)
    (
            len_type n,
      const void*    alpha_,
      const void*    beta_,
            bool     conj_A, void* A_, stride_type inc_A
    );

using trans_ft = void(*)
    (
            len_type m,
            len_type n,
      const void*    alpha_,
            bool     conj_A, const void* A_, stride_type rs_A, stride_type cs_A,
      const void*    beta_,
            bool     conj_B,       void* B_, stride_type rs_B, stride_type cs_B);

//
// Registration and intialization function prototypes.
//

#define plugin_tblis_params

#undef GENTCONF
#define GENTCONF( CONFIG, config ) \
\
void PASTEMAC(plugin_init_,config)(); \
void PASTEMAC(plugin_init_,config,BLIS_REF_SUFFIX)();

INSERT_GENTCONF

BLIS_EXPORT_BLIS err_t register_plugin();

}

#endif
