#include "util.hpp"
#include "dot.hpp"
#include "tblis/frame/1t/dense/dot.hpp"

namespace tblis
{
namespace internal
{

static
void dot_full(type_t type, const communicator& comm, const cntx_t* cntx,
              bool conj_A, const dpd_marray_view<char>& A,
              const dim_vector& idx_A_AB,
              bool conj_B, const dpd_marray_view<char>& B,
              const dim_vector& idx_B_AB,
              char* result)
{
    auto [A2, len_A2, stride_A2] = block_to_full(type, comm, cntx, A);
    auto [B2, len_B2, stride_B2] = block_to_full(type, comm, cntx, B);

    auto len_AB = stl_ext::select_from(len_A2, idx_A_AB);
    auto stride_A_AB = stl_ext::select_from(stride_A2, idx_A_AB);
    auto stride_B_AB = stl_ext::select_from(stride_B2, idx_B_AB);

    dot(type, comm, cntx, len_AB,
        conj_A, A2, stride_A_AB,
        conj_B, B2, stride_B_AB,
        result);

    if (comm.master())
    {
        delete[] A2;
        delete[] B2;
    }
}

static
void dot_block(type_t type, const communicator& comm, const cntx_t* cntx,
               bool conj_A, const dpd_marray_view<char>& A,
               const dim_vector& idx_A_AB,
               bool conj_B, const dpd_marray_view<char>& B,
               const dim_vector& idx_B_AB,
               char* result)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();
    const auto irrep = A.irrep();
    const auto ndim = A.dimension();

    stride_type nblock_AB = ipow(nirrep, ndim-1);

    std::array<len_vector,1> dense_len;
    std::array<stride_vector,1> dense_stride;
    dense_total_lengths_and_strides(dense_len, dense_stride, A, idx_A_AB);

    stride_type dense_size = stl_ext::prod(dense_len[0]);
    if (nblock_AB > 1)
        dense_size = std::max<stride_type>(1, dense_size/nirrep);

    atomic_accumulator local_result;

    comm.do_tasks_deferred(nblock_AB, dense_size*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
        {
            tasks.visit(block_AB,
            [&,block_AB](const communicator& subcomm)
            {
                irrep_vector irreps_A(ndim);
                irrep_vector irreps_B(ndim);

                assign_irreps(ndim, irrep, nirrep, block_AB,
                              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

                if (is_block_empty(A, irreps_A)) return;

                marray_view<char> local_A = A(irreps_A);
                marray_view<char> local_B = B(irreps_B);

                auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
                auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

                scalar block_result(0, type);

                dot(type, subcomm, cntx, len_AB,
                    conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_AB,
                    conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_AB,
                    block_result.raw());

                if (subcomm.master()) local_result += block_result;
            });
        }
    });

    reduce(type, comm, local_result);
    if (comm.master()) local_result.store(type, result);
}

void dot(type_t type, const communicator& comm, const cntx_t* cntx,
         bool conj_A, const dpd_marray_view<char>& A,
         const dim_vector& idx_A_AB,
         bool conj_B, const dpd_marray_view<char>& B,
         const dim_vector& idx_B_AB,
         char* result)
{
    if (A.irrep() != B.irrep())
    {
        if (comm.master()) memset(result, 0, type_size[type]);
        comm.barrier();
        return;
    }

    if (dpd_impl == FULL)
    {
        dot_full(type, comm, cntx,
                 conj_A, A, idx_A_AB,
                 conj_B, B, idx_B_AB,
                 result);
    }
    else
    {
        dot_block(type, comm, cntx,
                  conj_A, A, idx_A_AB,
                  conj_B, B, idx_B_AB,
                  result);
    }

    comm.barrier();
}

}
}
