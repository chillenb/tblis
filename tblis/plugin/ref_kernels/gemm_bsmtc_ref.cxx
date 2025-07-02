#include "../bli_plugin_tblis.h"
#include "../kernel.hpp"

namespace tblis
{

template <typename T>
void TBLIS_REF_KERNEL(gemm_bsmtc)
    (
            dim_t      m,
            dim_t      n,
            dim_t      k,
      const void*      alpha,
      const void*      a,
      const void*      b,
      const void*      beta0,
            void*      c0, stride_type rs_c, const stride_type* rscat_c0,
                           stride_type cs_c, const stride_type* cscat_c0,
      const auxinfo_t* auxinfo,
      const cntx_t*    cntx
    )
{
    auto cntl = static_cast<const cntl_t*>(bli_auxinfo_params(auxinfo));

    auto gemm_ukr = bli_gemm_var_cntl_ukr(cntl);
    auto row_pref = bli_gemm_var_cntl_row_pref(cntl);
    auto mr = bli_gemm_var_cntl_mr(cntl);
    auto nr = bli_gemm_var_cntl_nr(cntl);

    auto aux = *auxinfo;
    bli_auxinfo_set_params(bli_gemm_var_cntl_params(cntl), &aux);

    auto* TBLIS_RESTRICT c = static_cast<T*>(c0);
    auto* TBLIS_RESTRICT rscat_c = rscat_c0;
    auto* TBLIS_RESTRICT cscat_c = cscat_c0;

    if (rs_c && cs_c)
    {
        gemm_ukr(m, n, k,
                 alpha, a, b,
                 beta0, c + *rscat_c + *cscat_c, rs_c, cs_c,
                 &aux, cntx);
    }
    else
    {
        T beta = *static_cast<const T*>(beta0);
        T zero{};
        T ct[BLIS_STACK_BUF_MAX_SIZE / sizeof(T)]
            __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));

        auto rs_ct = row_pref ? nr : 1;
        auto cs_ct = row_pref ? 1 : mr;

        gemm_ukr(mr, nr, k,
                 alpha, a, b,
                 &zero, ct, rs_ct, cs_ct,
                 &aux, cntx);

        if (beta == zero)
        {
            for (len_type i = 0;i < m;i++)
                for (len_type j = 0;j < n;j++)
                    c[rscat_c[i] + cscat_c[j]] = ct[i*rs_ct + j*cs_ct];
        }
        else
        {
            for (len_type i = 0;i < m;i++)
                for (len_type j = 0;j < n;j++)
                    c[rscat_c[i] + cscat_c[j]] =
                        ct[i*rs_ct + j*cs_ct] + beta*c[rscat_c[i] + cscat_c[j]];
        }
    }
}

TBLIS_INIT_REF_KERNEL(gemm_bsmtc);

}
