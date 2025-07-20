#include "../bli_plugin_tblis.h"
#include "../kernel.hpp"
#include "tblis/frame/base/basic_types.h"

namespace tblis
{

inline void set0s_edge(dim_t i, dim_t m, dim_t j, dim_t n, float* p, stride_type ldp)
{
    bli_tset0s_edge(s, i, m, j, n, p, ldp);
}

inline void set0s_edge(dim_t i, dim_t m, dim_t j, dim_t n, double* p, stride_type ldp)
{
    bli_tset0s_edge(d, i, m, j, n, p, ldp);
}

inline void set0s_edge(dim_t i, dim_t m, dim_t j, dim_t n, scomplex* p, stride_type ldp)
{
    bli_tset0s_edge(c, i, m, j, n, reinterpret_cast<::scomplex*>(p), ldp);
}

inline void set0s_edge(dim_t i, dim_t m, dim_t j, dim_t n, dcomplex* p, stride_type ldp)
{
    bli_tset0s_edge(z, i, m, j, n, reinterpret_cast<::dcomplex*>(p), ldp);
}

template <typename T>
std::enable_if_t<!is_complex_v<T>,T> ri_to_ir(const T& x)
{
    return x;
}

template <typename T>
std::enable_if_t<is_complex_v<T>,T> ri_to_ir(const T& x)
{
    return T{-imag(x), real(x)};
}

template <typename T, typename U>
void TBLIS_REF_KERNEL(packm_bsmtc)
    (
            conj_t conjc,
            pack_t schema,
            dim_t  panel_dim,
            dim_t  panel_len,
            dim_t  panel_dim_max,
            dim_t  panel_len_max,
            dim_t  panel_bcast,
      const void*  kappa_,
      const void*  c_, const stride_type* rscat_c_,       stride_type  rbs_c,
                       const stride_type* cscat_c_, const stride_type* cbs_c_,
            void*  p_,       stride_type  ldp
    )
{
    using Ur = real_type_t<U>;
    constexpr auto KE = tblis::KE<U>::value;

    auto kappa = *static_cast<const U*>(kappa_);

    auto* TBLIS_RESTRICT c = static_cast<const T*>(c_);

    auto* TBLIS_RESTRICT rscat_c = rscat_c_;
    auto* TBLIS_RESTRICT cscat_c = cscat_c_;
    auto* TBLIS_RESTRICT cbs_c   = cbs_c_;

    if (schema == BLIS_PACKED_PANELS)
    {
        auto* TBLIS_RESTRICT p = static_cast<U*>(p_);

        auto body = [&,p](len_type cdim, len_type bcast, bool conjc) mutable
        {
            auto block = [&](len_type cdim, len_type bcast, len_type ldc, bool conjc, len_type KE)
            {
                if (ldc)
                {
                    auto cmk = c + *rscat_c + *cscat_c;
                    for (dim_t k = 0;k < KE;k++)
                    for (dim_t m = 0;m < cdim;m++)
                    for (dim_t d = 0;d < bcast;d++)
                        p[d + m*bcast + k*ldp] = kappa * tblis::conj(conjc, cmk[rbs_c*m + k*ldc]);
                }
                else
                {
                    auto cmk = c + *rscat_c;
                    for (dim_t k = 0;k < KE;k++)
                    for (dim_t m = 0;m < cdim;m++)
                    for (dim_t d = 0;d < bcast;d++)
                        p[d + m*bcast + k*ldp] = kappa * tblis::conj(conjc, cmk[rbs_c*m + cscat_c[k]]);
                }
            };

            auto scat = [&](len_type cdim, len_type bcast, bool conjc, len_type KE)
            {
                for (dim_t k = 0;k < KE;k++)
                for (dim_t m = 0;m < cdim;m++)
                for (dim_t d = 0;d < bcast;d++)
                    p[d + m*bcast + k*ldp] = kappa * tblis::conj(conjc, c[rscat_c[m] + cscat_c[k]]);
            };

            if (rbs_c)
            {
                dim_t k = 0;
                while (k <= panel_len-KE)
                {
                    block(cdim, bcast, *cbs_c, conjc, KE);

                    cscat_c += KE;
                    p += ldp*KE;
                    k += KE;
                    cbs_c++;
                }

                block(cdim, bcast, *cbs_c, conjc, panel_len-k);
            }
            else
            {
                dim_t k = 0;
                while (k <= panel_len-KE)
                {
                    scat(cdim, bcast, conjc, KE);

                    cscat_c += KE;
                    p += ldp*KE;
                    k += KE;
                }

                scat(cdim, bcast, conjc, panel_len-k);
            }
        };

        if (conjc && is_complex_v<T>)
        {
            body(panel_dim, panel_bcast, true);
        }
        else
        {
            body(panel_dim, panel_bcast, false);
        }

        set0s_edge
        (
            panel_dim*panel_bcast, panel_dim_max*panel_bcast,
            panel_len, panel_len_max,
            p, ldp
        );
    }
    else if (is_complex_v<T> && schema == BLIS_PACKED_PANELS_1R)
    {
        auto* TBLIS_RESTRICT pr = static_cast<Ur*>(p_);
        auto* TBLIS_RESTRICT pi = static_cast<Ur*>(p_) + ldp;

        auto body = [&,pr,pi](len_type cdim, len_type bcast, bool conjc) mutable
        {
            auto block = [&](len_type cdim, len_type bcast, len_type ldc, bool conjc, len_type KE)
            {
                if (ldc)
                {
                    auto cmk = c + *rscat_c + *cscat_c;
                    for (dim_t k = 0;k < KE;k++)
                    for (dim_t m = 0;m < cdim;m++)
                    for (dim_t d = 0;d < bcast;d++)
                    {
                        auto tmp = kappa * tblis::conj(conjc, cmk[rbs_c*m + k*ldc]);
                        pr[d + m*bcast + k*2*ldp] = real(tmp);
                        pi[d + m*bcast + k*2*ldp] = imag(tmp);
                    }
                }
                else
                {
                    auto cmk = c + *rscat_c;
                    for (dim_t k = 0;k < KE;k++)
                    for (dim_t m = 0;m < cdim;m++)
                    for (dim_t d = 0;d < bcast;d++)
                    {
                        auto tmp = kappa * tblis::conj(conjc, cmk[rbs_c*m + cscat_c[k]]);
                        pr[d + m*bcast + k*2*ldp] = real(tmp);
                        pi[d + m*bcast + k*2*ldp] = imag(tmp);
                    }
                }
            };

            auto scat = [&](len_type cdim, len_type bcast, bool conjc, len_type KE)
            {
                for (dim_t k = 0;k < KE;k++)
                for (dim_t m = 0;m < cdim;m++)
                for (dim_t d = 0;d < bcast;d++)
                {
                    auto tmp = kappa * tblis::conj(conjc, c[rscat_c[m] + cscat_c[k]]);
                    pr[d + m*bcast + k*2*ldp] = real(tmp);
                    pi[d + m*bcast + k*2*ldp] = imag(tmp);
                }
            };

            if (rbs_c)
            {
                dim_t k = 0;
                while (k <= panel_len-KE)
                {
                    block(cdim, bcast, *cbs_c, conjc, KE);

                    cscat_c += KE;
                    pr += 2*ldp*KE;
                    pi += 2*ldp*KE;
                    k += KE;
                    cbs_c++;
                }

                block(cdim, bcast, *cbs_c, conjc, panel_len-k);
            }
            else
            {
                dim_t k = 0;
                while (k <= panel_len-KE)
                {
                    scat(cdim, bcast, conjc, KE);

                    cscat_c += KE;
                    pr += 2*ldp*KE;
                    pi += 2*ldp*KE;
                    k += KE;
                }

                scat(cdim, bcast, conjc, panel_len-k);
            }
        };

        if (conjc && is_complex_v<T>)
        {
            body(panel_dim, panel_bcast, true);
        }
        else
        {
            body(panel_dim, panel_bcast, false);
        }

        set0s_edge
        (
            panel_dim*panel_bcast, panel_dim_max*panel_bcast,
            2*panel_len, 2*panel_len_max,
            pr, ldp
        );
    }
    else if (is_complex_v<T> && schema == BLIS_PACKED_PANELS_1E)
    {
        auto* TBLIS_RESTRICT pri = static_cast<U*>(p_);
        auto* TBLIS_RESTRICT pir = static_cast<U*>(p_) + ldp/2;

        auto body = [&,pri,pir](len_type cdim, len_type bcast, bool conjc) mutable
        {
            auto block = [&](len_type cdim, len_type bcast, len_type ldc, bool conjc, len_type KE)
            {
                if (ldc)
                {
                    auto cmk = c + *rscat_c + *cscat_c;
                    for (dim_t k = 0;k < KE;k++)
                    for (dim_t m = 0;m < cdim;m++)
                    for (dim_t d = 0;d < bcast;d++)
                    {
                        auto tmp = kappa * tblis::conj(conjc, cmk[rbs_c*m + k*ldc]);
                        pri[d + m*bcast + k*ldp] = tmp;
                        pir[d + m*bcast + k*ldp] = ri_to_ir(tmp);
                    }
                }
                else
                {
                    auto cmk = c + *rscat_c;
                    for (dim_t k = 0;k < KE;k++)
                    for (dim_t m = 0;m < cdim;m++)
                    for (dim_t d = 0;d < bcast;d++)
                    {
                        auto tmp = kappa * tblis::conj(conjc, cmk[rbs_c*m + cscat_c[k]]);
                        pri[d + m*bcast + k*ldp] = tmp;
                        pir[d + m*bcast + k*ldp] = ri_to_ir(tmp);
                    }
                }
            };

            auto scat = [&](len_type cdim, len_type bcast, bool conjc, len_type KE)
            {
                for (dim_t k = 0;k < KE;k++)
                for (dim_t m = 0;m < cdim;m++)
                for (dim_t d = 0;d < bcast;d++)
                {
                    auto tmp = kappa * tblis::conj(conjc, c[rscat_c[m] + cscat_c[k]]);
                    pri[d + m*bcast + k*ldp] = tmp;
                    pir[d + m*bcast + k*ldp] = ri_to_ir(tmp);
                }
            };

            if (rbs_c)
            {
                dim_t k = 0;
                while (k <= panel_len-KE)
                {
                    block(cdim, bcast, *cbs_c, conjc, KE);

                    cscat_c += KE;
                    pri += ldp*KE;
                    pir += ldp*KE;
                    k += KE;
                    cbs_c++;
                }

                block(cdim, bcast, *cbs_c, conjc, panel_len-k);
            }
            else
            {
                dim_t k = 0;
                while (k <= panel_len-KE)
                {
                    scat(cdim, bcast, conjc, KE);

                    cscat_c += KE;
                    pri += ldp*KE;
                    pir += ldp*KE;
                    k += KE;
                }

                scat(cdim, bcast, conjc, panel_len-k);
            }
        };

        if (conjc && is_complex_v<T>)
        {
            body(panel_dim, panel_bcast, true);
        }
        else
        {
            body(panel_dim, panel_bcast, false);
        }

        set0s_edge
        (
            panel_dim*panel_bcast, panel_dim_max*panel_bcast,
            2*panel_len, 2*panel_len_max,
            pri, ldp/2
        );
    }
    else if (is_complex_v<T> && schema == BLIS_PACKED_PANELS_RO)
    {
        auto* TBLIS_RESTRICT pr = static_cast<Ur*>(p_);

        auto body = [&,pr](len_type cdim, len_type bcast, bool conjc) mutable
        {
            auto block = [&](len_type cdim, len_type bcast, len_type ldc, bool conjc, len_type KE)
            {
                if (ldc)
                {
                    auto cmk = c + *rscat_c + *cscat_c;
                    for (dim_t k = 0;k < KE;k++)
                    for (dim_t m = 0;m < cdim;m++)
                    for (dim_t d = 0;d < bcast;d++)
                        pr[d + m*bcast + k*ldp] = real(kappa * tblis::conj(conjc, cmk[rbs_c*m + k*ldc]));
                }
                else
                {
                    auto cmk = c + *rscat_c;
                    for (dim_t k = 0;k < KE;k++)
                    for (dim_t m = 0;m < cdim;m++)
                    for (dim_t d = 0;d < bcast;d++)
                        pr[d + m*bcast + k*ldp] = real(kappa * tblis::conj(conjc, cmk[rbs_c*m + cscat_c[k]]));
                }
            };

            auto scat = [&](len_type cdim, len_type bcast, bool conjc, len_type KE)
            {
                for (dim_t k = 0;k < KE;k++)
                for (dim_t m = 0;m < cdim;m++)
                for (dim_t d = 0;d < bcast;d++)
                    pr[d + m*bcast + k*ldp] = real(kappa * tblis::conj(conjc, c[rscat_c[m] + cscat_c[k]]));
            };

            if (rbs_c)
            {
                dim_t k = 0;
                while (k <= panel_len-KE)
                {
                    block(cdim, bcast, *cbs_c, conjc, KE);

                    cscat_c += KE;
                    pr += ldp*KE;
                    k += KE;
                    cbs_c++;
                }

                block(cdim, bcast, *cbs_c, conjc, panel_len-k);
            }
            else
            {
                dim_t k = 0;
                while (k <= panel_len-KE)
                {
                    scat(cdim, bcast, conjc, KE);

                    cscat_c += KE;
                    pr += ldp*KE;
                    k += KE;
                }

                scat(cdim, bcast, conjc, panel_len-k);
            }
        };

        if (conjc && is_complex_v<T>)
        {
            body(panel_dim, panel_bcast, true);
        }
        else
        {
            body(panel_dim, panel_bcast, false);
        }

        set0s_edge
        (
            panel_dim*panel_bcast, panel_dim_max*panel_bcast,
            panel_len, panel_len_max,
            pr, ldp
        );
    }
}

TBLIS_INIT_REF_KERNEL2(packm_bsmtc);

}
