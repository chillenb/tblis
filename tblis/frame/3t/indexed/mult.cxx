#include "mult.hpp"
#include "tblis/frame/1t/indexed/scale.hpp"
#include "tblis/frame/1t/indexed/set.hpp"
#include "tblis/frame/1t/indexed/util.hpp"
#include "tblis/frame/3t/dense/mult.hpp"

#include "tblis/frame/base/tensor.hpp"

namespace tblis
{
namespace internal
{

static
void mult_full(type_t type, const communicator& comm, const cntx_t* cntx,
               const scalar& alpha, bool conj_A, const indexed_marray_view<char>& A,
               const dim_vector& idx_A_AB,
               const dim_vector& idx_A_AC,
               const dim_vector& idx_A_ABC,
                                    bool conj_B, const indexed_marray_view<char>& B,
               const dim_vector& idx_B_AB,
               const dim_vector& idx_B_BC,
               const dim_vector& idx_B_ABC,
                                                 const indexed_marray_view<char>& C,
               const dim_vector& idx_C_AC,
               const dim_vector& idx_C_BC,
               const dim_vector& idx_C_ABC)
{
    auto [A2, len_A2, stride_A2] = block_to_full(type, comm, cntx, A);
    auto [B2, len_B2, stride_B2] = block_to_full(type, comm, cntx, B);
    auto [C2, len_C2, stride_C2] = block_to_full(type, comm, cntx, C);

    auto len_AB = stl_ext::select_from(len_A2, idx_A_AB);
    auto len_AC = stl_ext::select_from(len_C2, idx_C_AC);
    auto len_BC = stl_ext::select_from(len_C2, idx_C_BC);
    auto len_ABC = stl_ext::select_from(len_C2, idx_C_ABC);
    auto stride_A_AB = stl_ext::select_from(stride_A2, idx_A_AB);
    auto stride_A_AC = stl_ext::select_from(stride_A2, idx_A_AC);
    auto stride_B_AB = stl_ext::select_from(stride_B2, idx_B_AB);
    auto stride_B_BC = stl_ext::select_from(stride_B2, idx_B_BC);
    auto stride_C_AC = stl_ext::select_from(stride_C2, idx_C_AC);
    auto stride_C_BC = stl_ext::select_from(stride_C2, idx_C_BC);
    auto stride_A_ABC = stl_ext::select_from(stride_A2, idx_A_ABC);
    auto stride_B_ABC = stl_ext::select_from(stride_B2, idx_B_ABC);
    auto stride_C_ABC = stl_ext::select_from(stride_C2, idx_C_ABC);

    scalar zero(0.0, type);

    mult(type, comm, cntx, len_AB, len_AC, len_BC, len_ABC,
         alpha, conj_A, A2, stride_A_AB, stride_A_AC, stride_A_ABC,
                conj_B, B2, stride_B_AB, stride_B_BC, stride_B_ABC,
          zero,  false, C2, stride_C_AC, stride_C_BC, stride_C_ABC);

    full_to_block(type, comm, cntx, C2, len_C2, stride_C2, C);

    if (comm.master())
    {
        delete[] A2;
        delete[] B2;
        delete[] C2;
    }
}

static
void contract_block(type_t type, const communicator& comm, const cntx_t* cntx,
                    const scalar& alpha,
                    bool conj_A, const indexed_marray_view<char>& A,
                    dim_vector idx_A_AB,
                    dim_vector idx_A_AC,
                    bool conj_B, const indexed_marray_view<char>& B,
                    dim_vector idx_B_AB,
                    dim_vector idx_B_BC,
                                 const indexed_marray_view<char>& C,
                    dim_vector idx_C_AC,
                    dim_vector idx_C_BC)
{
    const len_type ts = type_size[type];

    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    group_indices<2> indices_A(type, A, group_AC, 0, group_AB, 0);
    group_indices<2> indices_B(type, B, group_BC, 0, group_AB, 1);
    group_indices<2> indices_C(type, C, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    scalar one(1.0, type);

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_C = 0;

    comm.do_tasks_deferred(nidx_C, stl_ext::prod(group_AB.dense_len)*
                                   stl_ext::prod(group_AC.dense_len)*
                                   stl_ext::prod(group_BC.dense_len)*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<true, true>(idx_A, nidx_A, indices_A, 0,
                                  idx_C, nidx_C, indices_C, 0,
        [&](stride_type next_A, stride_type next_C)
        {
            stride_type idx_B = 0;

            for_each_match<true, false>(idx_B, nidx_B, indices_B, 0,
                                       idx_C, next_C, indices_C, 1,
            [&](stride_type next_B)
            {
                if (indices_C[idx_C].factor.is_zero()) return;

                tasks.visit(idx++,
                [&,idx_A,idx_B,idx_C,next_A,next_B]
                (const communicator& subcomm)
                {
                    auto local_idx_A = idx_A;
                    auto local_idx_B = idx_B;

                    stride_type off_A_AC, off_C_AC;
                    get_local_offset(indices_A[local_idx_A].idx[0], group_AC,
                                     off_A_AC, 0, off_C_AC, 1);

                    stride_type off_B_BC, off_C_BC;
                    get_local_offset(indices_B[local_idx_B].idx[0], group_BC,
                                     off_B_BC, 0, off_C_BC, 1);

                    auto data_C = C.data(0) + (indices_C[idx_C].offset + off_C_AC + off_C_BC)*ts;

                    for_each_match<false, false>(local_idx_A, next_A, indices_A, 1,
                                                local_idx_B, next_B, indices_B, 1,
                    [&]
                    {
                        auto factor = alpha*indices_A[local_idx_A].factor*
                                            indices_B[local_idx_B].factor*
                                            indices_C[idx_C].factor;
                        if (factor.is_zero()) return;

                        stride_type off_A_AB, off_B_AB;
                        get_local_offset(indices_A[local_idx_A].idx[1], group_AB,
                                         off_A_AB, 0, off_B_AB, 1);

                        auto data_A = A.data(0) + (indices_A[local_idx_A].offset + off_A_AB + off_A_AC)*ts;
                        auto data_B = B.data(0) + (indices_B[local_idx_B].offset + off_B_AB + off_B_BC)*ts;

                        mult(type, subcomm, cntx,
                             group_AB.dense_len,
                             group_AC.dense_len,
                             group_BC.dense_len, {},
                             factor, conj_A, data_A, group_AB.dense_stride[0],
                                                     group_AC.dense_stride[0], {},
                                     conj_B, data_B, group_AB.dense_stride[1],
                                                     group_BC.dense_stride[0], {},
                                one,  false, data_C, group_AC.dense_stride[1],
                                                     group_BC.dense_stride[1], {});
                    });
                });
            });
        });
    });
}

static
void mult_block(type_t type, const communicator& comm, const cntx_t* cntx,
                const scalar& alpha,
                bool conj_A, const indexed_marray_view<char>& A,
                dim_vector idx_A_AB,
                dim_vector idx_A_AC,
                dim_vector idx_A_ABC,
                bool conj_B, const indexed_marray_view<char>& B,
                dim_vector idx_B_AB,
                dim_vector idx_B_BC,
                dim_vector idx_B_ABC,
                             const indexed_marray_view<char>& C,
                dim_vector idx_C_AC,
                dim_vector idx_C_BC,
                dim_vector idx_C_ABC)
{
    const len_type ts = type_size[type];

    index_group<3> group_ABC(A, idx_A_ABC, B, idx_B_ABC, C, idx_C_ABC);
    index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    group_indices<3> indices_A(type, A, group_ABC, 0, group_AC, 0, group_AB, 0);
    group_indices<3> indices_B(type, B, group_ABC, 1, group_BC, 0, group_AB, 1);
    group_indices<3> indices_C(type, C, group_ABC, 2, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    scalar one(1.0, type);

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B0 = 0;
    stride_type idx_C = 0;

    comm.do_tasks_deferred(nidx_C, stl_ext::prod(group_AB.dense_len)*
                                   stl_ext::prod(group_AC.dense_len)*
                                   stl_ext::prod(group_BC.dense_len)*
                                   stl_ext::prod(group_ABC.dense_len)*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<true, true, true>(idx_A,  nidx_A, indices_A, 0,
                                        idx_B0, nidx_B, indices_B, 0,
                                        idx_C,  nidx_C, indices_C, 0,
        [&](stride_type next_A_ABC, stride_type next_B_ABC, stride_type next_C_ABC)
        {
            for_each_match<true, true>(idx_A, next_A_ABC, indices_A, 1,
                                      idx_C, next_C_ABC, indices_C, 1,
            [&](stride_type next_A_AB, stride_type next_C_AC)
            {
                stride_type idx_B = idx_B0;

                for_each_match<true, false>(idx_B, next_B_ABC, indices_B, 1,
                                           idx_C,  next_C_AC, indices_C, 2,
                [&](stride_type next_B_AB)
                {
                    if (indices_C[idx_C].factor.is_zero()) return;

                    tasks.visit(idx++,
                    [&,idx_A,idx_B,idx_C,next_A_AB,next_B_AB]
                    (const communicator& subcomm)
                    {
                        auto local_idx_A = idx_A;
                        auto local_idx_B = idx_B;

                        stride_type off_A_ABC, off_B_ABC, off_C_ABC;
                        get_local_offset(indices_A[local_idx_A].idx[0], group_ABC,
                                         off_A_ABC, 0, off_B_ABC, 1, off_C_ABC, 2);

                        stride_type off_A_AC, off_C_AC;
                        get_local_offset(indices_A[local_idx_A].idx[1], group_AC,
                                         off_A_AC, 0, off_C_AC, 1);

                        stride_type off_B_BC, off_C_BC;
                        get_local_offset(indices_B[local_idx_B].idx[1], group_BC,
                                         off_B_BC, 0, off_C_BC, 1);

                        auto data_C = C.data(0) + (indices_C[idx_C].offset + off_C_AC + off_C_BC + off_C_ABC)*ts;

                        for_each_match<false, false>(local_idx_A, next_A_AB, indices_A, 2,
                                                    local_idx_B, next_B_AB, indices_B, 2,
                        [&]
                        {
                            auto factor = alpha*indices_A[local_idx_A].factor*
                                                indices_B[local_idx_B].factor*
                                                indices_C[idx_C].factor;
                            if (factor.is_zero()) return;

                            stride_type off_A_AB, off_B_AB;
                            get_local_offset(indices_A[local_idx_A].idx[2], group_AB,
                                             off_A_AB, 0, off_B_AB, 1);

                            auto data_A = A.data(0) + (indices_A[local_idx_A].offset + off_A_AB + off_A_AC + off_A_ABC)*ts;
                            auto data_B = B.data(0) + (indices_B[local_idx_B].offset + off_B_AB + off_B_BC + off_B_ABC)*ts;

                            mult(type, subcomm, cntx,
                                 group_AB.dense_len,
                                 group_AC.dense_len,
                                 group_BC.dense_len,
                                 group_ABC.dense_len,
                                 factor, conj_A, data_A, group_AB.dense_stride[0],
                                                         group_AC.dense_stride[0],
                                                         group_ABC.dense_stride[0],
                                         conj_B, data_B, group_AB.dense_stride[1],
                                                         group_BC.dense_stride[0],
                                                         group_ABC.dense_stride[1],
                                    one,  false, data_C, group_AC.dense_stride[1],
                                                         group_BC.dense_stride[1],
                                                         group_ABC.dense_stride[2]);
                        });
                    });
                });
            });
        });
    });
}

void mult(type_t type, const communicator& comm, const cntx_t* cntx,
          const scalar& alpha, bool conj_A, const indexed_marray_view<char>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
                               bool conj_B, const indexed_marray_view<char>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          const scalar&  beta, bool conj_C, const indexed_marray_view<char>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC)
{
    if (beta.is_zero())
    {
        set(type, comm, cntx, beta, C, range(C.dimension()));
    }
    else if (!beta.is_one() || (beta.is_complex() && conj_C))
    {
        scale(type, comm, cntx, beta, conj_C, C, range(C.dimension()));
    }

    if (dpd_impl == FULL)
    {
        mult_full(type, comm, cntx, alpha,
                  conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                  conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                          C, idx_C_AC, idx_C_BC, idx_C_ABC);
    }
    else if (idx_C_ABC.empty())
    {
        contract_block(type, comm, cntx,
                       alpha, conj_A, A, idx_A_AB, idx_A_AC,
                              conj_B, B, idx_B_AB, idx_B_BC,
                                      C, idx_C_AC, idx_C_BC);
    }
    else
    {
        mult_block(type, comm, cntx,
                   alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                          conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                                  C, idx_C_AC, idx_C_BC, idx_C_ABC);
    }

    comm.barrier();
}

}
}
