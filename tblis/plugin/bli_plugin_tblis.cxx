#include "bli_plugin_tblis.h"

namespace tblis
{

kerid_t GEMM_BSMTC_UKR = -1;
kerid_t PACKM_BSMTC_UKR = -1;
kerid_t MULT_KER = -1;
kerid_t REDUCE_KER = -1;
kerid_t SHIFT_KER = -1;
kerid_t TRANS_KER = -1;
kerid_t MRT_BSZ = -1;
kerid_t NRT_BSZ = -1;
kerid_t KE_BSZ = -1;

err_t register_plugin()
{
    bli_init();

    if (auto err = bli_gks_register_ukr2(&PACKM_BSMTC_UKR); err != BLIS_SUCCESS) return err;
    if (auto err = bli_gks_register_ukr(&GEMM_BSMTC_UKR); err != BLIS_SUCCESS) return err;

    if (auto err = bli_gks_register_ukr(&MULT_KER); err != BLIS_SUCCESS) return err;
    if (auto err = bli_gks_register_ukr(&REDUCE_KER); err != BLIS_SUCCESS) return err;
    if (auto err = bli_gks_register_ukr(&SHIFT_KER); err != BLIS_SUCCESS) return err;
    if (auto err = bli_gks_register_ukr(&TRANS_KER); err != BLIS_SUCCESS) return err;

    if (auto err = bli_gks_register_blksz(&MRT_BSZ); err != BLIS_SUCCESS) return err;
    if (auto err = bli_gks_register_blksz(&NRT_BSZ); err != BLIS_SUCCESS) return err;
    if (auto err = bli_gks_register_blksz(&KE_BSZ); err != BLIS_SUCCESS) return err;

    //
    // Initialize the context for each enabled sub-configuration.
    //

    #undef GENTCONF
    #define GENTCONF( CONFIG, config ) \
    PASTEMAC(plugin_init_,config,BLIS_REF_SUFFIX)();

    INSERT_GENTCONF

    return BLIS_SUCCESS;
}

}

