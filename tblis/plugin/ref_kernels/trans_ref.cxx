#include "../bli_plugin_tblis.h"
#include "../kernel.hpp"

namespace tblis
{

template <typename T>
void TBLIS_REF_KERNEL(trans)
    (
            len_type m,
            len_type n,
      const void*    alpha_,
            bool     conj_A, const void* A_, stride_type rs_A, stride_type cs_A,
      const void*    beta_,
            bool     conj_B,       void* B_, stride_type rs_B, stride_type cs_B
    )
{
    constexpr auto MR = tblis::MRT<T>::value;
    constexpr auto NR = tblis::NRT<T>::value;

    T alpha = *static_cast<const T*>(alpha_);
    T beta  = *static_cast<const T*>(beta_ );

    const T* TBLIS_RESTRICT A = static_cast<const T*>(A_);
          T* TBLIS_RESTRICT B = static_cast<      T*>(B_);

    auto block = [&](bool conja, bool conjb, bool bz, len_type m, len_type n,
                     stride_type rs_A, stride_type cs_A, stride_type rs_B, stride_type cs_B)
    {
        for (len_type i = 0;i < m;i++)
        for (len_type j = 0;j < n;j++)
        {
           if (bz) B[j*cs_B + i*rs_B] = alpha*tblis::conj(conja, A[i*rs_A + j*cs_A]);
           else    B[j*cs_B + i*rs_B] = alpha*tblis::conj(conja, A[i*rs_A + j*cs_A]) +
                                        beta *tblis::conj(conjb, B[j*cs_B + i*rs_B]);
        }
    };

    auto body = [&](bool conja, bool conjb, bool bz, len_type m, len_type n)
    {
        if      (rs_B == 1 && rs_A == 1) block(conja, conjb, bz, m, n,    1, cs_A,    1, cs_B);
        else if (cs_B == 1 && cs_A == 1) block(conja, conjb, bz, m, n, rs_A,    1, rs_B,    1);
        else                             block(conja, conjb, bz, m, n, rs_A, cs_A, rs_B, cs_B);
    };

    if (m == MR && n == NR)
    {
        if (beta == T(0))
        {
            if (is_complex_v<T> && conj_A) body( true, false, true, MR, NR);
            else                           body(false, false, true, MR, NR);
        }
        else
        {
            if (is_complex_v<T> && conj_A && conj_B) body( true,  true, false, MR, NR);
            if (is_complex_v<T> && conj_A          ) body( true, false, false, MR, NR);
            if (is_complex_v<T>           && conj_B) body(false,  true, false, MR, NR);
            else                                     body(false, false, false, MR, NR);
        }
    }
    else
    {
        if (beta == T(0))
        {
            if (is_complex_v<T> && conj_A) body( true, false, true, m, n);
            else                           body(false, false, true, m, n);
        }
        else
        {
            if (is_complex_v<T> && conj_A && conj_B) body( true,  true, false, m, n);
            if (is_complex_v<T> && conj_A          ) body( true, false, false, m, n);
            if (is_complex_v<T>           && conj_B) body(false,  true, false, m, n);
            else                                     body(false, false, false, m, n);
        }
    }
}

TBLIS_INIT_REF_KERNEL(trans)

}
