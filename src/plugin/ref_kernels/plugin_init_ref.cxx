#include "../bli_plugin_tblis.h"
#include "../kernel.hpp"

namespace tblis
{

// -----------------------------------------------------------------------------

void PASTEMAC(plugin_init,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX)()
{
    auto cntx = const_cast<cntx_t*>(bli_gks_lookup_id(PASTECH(BLIS_ARCH,BLIS_CNAME_UPPER_INFIX)));

    bli_cntx_set_ukr2(PACKM_BSMTC_UKR, &TBLIS_REF_KERNEL_FPA(packm_bsmtc), cntx);
    bli_cntx_set_ukr(GEMM_BSMTC_UKR, &TBLIS_REF_KERNEL_FPA(gemm_bsmtc), cntx);

    bli_cntx_set_ukr(MULT_KER, &TBLIS_REF_KERNEL_FPA(mult), cntx);
    bli_cntx_set_ukr(REDUCE_KER, &TBLIS_REF_KERNEL_FPA(reduce), cntx);
    bli_cntx_set_ukr(SHIFT_KER, &TBLIS_REF_KERNEL_FPA(shift), cntx);
    bli_cntx_set_ukr(TRANS_KER, &TBLIS_REF_KERNEL_FPA(trans), cntx);

    blksz_t mrt;
    bli_blksz_init_easy(&mrt, BLIS_MRT_s, BLIS_MRT_d, BLIS_MRT_c, BLIS_MRT_z);
    bli_cntx_set_blksz((bszid_t)MRT_BSZ, &mrt, BLIS_NO_PART, cntx);

    blksz_t nrt;
    bli_blksz_init_easy(&nrt, BLIS_NRT_s, BLIS_NRT_d, BLIS_NRT_c, BLIS_NRT_z);
    bli_cntx_set_blksz((bszid_t)NRT_BSZ, &nrt, BLIS_NO_PART, cntx);

    blksz_t ke;
    bli_blksz_init_easy(&ke, BLIS_KE_s, BLIS_KE_d, BLIS_KE_c, BLIS_KE_z);
    bli_cntx_set_blksz((bszid_t)KE_BSZ, &ke, BLIS_NO_PART, cntx);
}

}
