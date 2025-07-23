#include "scale.hpp"

#include "tblis/frame/base/tensor.hpp"

#include "tblis/plugin/bli_plugin_tblis.h"

namespace tblis
{
namespace internal
{

void scale(type_t type, const communicator& comm, const cntx_t* cntx,
           const len_vector& len_A, const scalar& alpha,
           bool conj_A, char* A, const stride_vector& stride_A)
{
    bli_init();

    bool empty = len_A.size() == 0;

    const len_type ts = type_size[type];

    len_type n0 = (empty ? 1 : len_A[0]);
    len_vector len1(len_A.begin() + !empty, len_A.end());
    len_type n1 = stl_ext::prod(len1);

    stride_type stride0 = (empty ? 1 : stride_A[0]);
    len_vector stride1;
    for (auto i : range(1,len_A.size())) stride1.push_back(stride_A[i]*ts);

    auto scale_ukr = reinterpret_cast<scal2v_ker_ft>(bli_cntx_get_ukr_dt((num_t)type, BLIS_SCAL2V_KER, cntx));

    comm.distribute_over_threads(n0, n1,
    [&](len_type n0_min, len_type n0_max, len_type n1_min, len_type n1_max)
    {
        auto A1 = A;

        viterator<1> iter_A(len1, stride1);
        iter_A.position(n1_min, A1);

        A1 += n0_min*stride0*ts;

        for (len_type i = n1_min;i < n1_max;i++)
        {
            iter_A.next(A1);
            // Use the `scal2v` kernel rather than `scalv` because the `conj` parameter in the latter
            // only applies to `alpha` and not the elements of `A`.
            scale_ukr(conj_A ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                      n0_max-n0_min, &alpha, A1, stride0, A1, stride0, cntx);
        }
    });

    comm.barrier();
}

}
}
