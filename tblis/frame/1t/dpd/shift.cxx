#include "util.hpp"
#include "shift.hpp"
#include "tblis/frame/1t/dense/shift.hpp"

namespace tblis
{
namespace internal
{

void shift(type_t type, const communicator& comm, const cntx_t* cntx,
           const scalar& alpha, const scalar& beta,
           bool conj_A, const dpd_marray_view<char>& A, const dim_vector& idx_A)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();
    const auto irrep = A.irrep();
    const auto ndim = A.dimension();

    stride_type nblock = ipow(nirrep, ndim-1);

    irrep_vector irreps(ndim);

    for (stride_type block = 0;block < nblock;block++)
    {
        assign_irreps(ndim, irrep, nirrep, block, irreps, idx_A);

        if (is_block_empty(A, irreps)) continue;

        marray_view<char> local_A = A(irreps);

        shift(type, comm, cntx, local_A.lengths(), alpha, beta, conj_A,
              A.data() + (local_A.data()-A.data())*ts, local_A.strides());
    }
}

}
}
