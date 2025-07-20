#include "util.hpp"
#include "dot.hpp"
#include "tblis/frame/1t/dense/dot.hpp"

namespace tblis
{
namespace internal
{

static
void dot_full(type_t type, const communicator& comm, const cntx_t* cntx,
              bool conj_A, const indexed_dpd_marray_view<char>& A,
              const dim_vector& idx_A_AB,
              bool conj_B, const indexed_dpd_marray_view<char>& B,
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
               bool conj_A, const indexed_dpd_marray_view<char>& A,
               const dim_vector& idx_A_AB,
               bool conj_B, const indexed_dpd_marray_view<char>& B,
               const dim_vector& idx_B_AB,
               char* result)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);

    auto irrep_AB = A.irrep();
    for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;

    if (group_AB.dense_ndim == 0 && irrep_AB != 0)
    {
        if (comm.master()) memset(result, 0, ts);
        return;
    }

    group_indices<1> indices_A(type, A, group_AB, 0);
    group_indices<1> indices_B(type, B, group_AB, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();

    auto dpd_A = A[0];
    auto dpd_B = B[0];

    atomic_accumulator local_result;

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B = 0;

    comm.do_tasks_deferred(std::min(nidx_A, nidx_B)*group_AB.dense_nblock,
                           group_AB.dense_size*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<false, false>(idx_A, nidx_A, indices_A, 0,
                                    idx_B, nidx_B, indices_B, 0,
        [&]
        {
            auto factor = indices_A[idx_A].factor*indices_B[idx_B].factor;
            if (factor.is_zero()) return;

            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
            {
                tasks.visit(idx++,
                [&,factor,idx_A,idx_B,block_AB](const communicator& subcomm)
                {
                    auto local_irreps_A = irreps_A;
                    auto local_irreps_B = irreps_B;

                    assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                  local_irreps_A, group_AB.dense_idx[0],
                                  local_irreps_B, group_AB.dense_idx[1]);

                    if (is_block_empty(dpd_A, local_irreps_A)) return;

                    marray_view<char> local_A = dpd_A(local_irreps_A);
                    marray_view<char> local_B = dpd_B(local_irreps_B);

                    len_vector len_AB;
                    stride_vector stride_A_AB, stride_B_AB;
                    stride_type off_A_AB, off_B_AB;
                    get_local_geometry(indices_A[idx_A].idx[0], group_AB, len_AB,
                                       local_A, stride_A_AB, 0,
                                       local_B, stride_B_AB, 1);
                    get_local_offset(indices_A[idx_A].idx[0], group_AB,
                                     local_A, off_A_AB, 0,
                                     local_B, off_B_AB, 1);

                    auto data_A = dpd_A.data() + (local_A.data() - dpd_A.data() +
                        indices_A[idx_A].offset + off_A_AB)*ts;
                    auto data_B = dpd_B.data() + (local_B.data() - dpd_B.data() +
                        indices_B[idx_B].offset + off_B_AB)*ts;

                    scalar block_result(0, type);

                    dot(type, subcomm, cntx, len_AB,
                        conj_A, data_A, stride_A_AB,
                        conj_B, data_B, stride_B_AB,
                        block_result.raw());

                    if (subcomm.master()) local_result += factor*block_result;
                });
            }
        });
    });

    reduce(type, comm, local_result);
    if (comm.master()) local_result.store(type, result);
}

void dot(type_t type, const communicator& comm, const cntx_t* cntx,
         bool conj_A, const indexed_dpd_marray_view<char>& A,
         const dim_vector& idx_A_AB,
         bool conj_B, const indexed_dpd_marray_view<char>& B,
         const dim_vector& idx_B_AB,
         char* result)
{
    if (A.irrep() != B.irrep())
    {
        if (comm.master()) memset(result, 0, type_size[type]);
        comm.barrier();
        return;
    }

    for (auto i : range(idx_A_AB.size()))
    {
        if (idx_A_AB[i] >= A.dense_dimension() &&
            idx_B_AB[i] >= B.dense_dimension())
        {
            if (A.indexed_irrep(idx_A_AB[i] - A.dense_dimension()) !=
                B.indexed_irrep(idx_B_AB[i] - B.dense_dimension()))
            {
                if (comm.master()) memset(result, 0, type_size[type]);
                comm.barrier();
                return;
            }
        }
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
