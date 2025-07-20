#include "util.hpp"
#include "add.hpp"
#include "tblis/frame/1t/dense/add.hpp"
#include "tblis/frame/1t/dense/scale.hpp"
#include "tblis/frame/1t/dense/set.hpp"

namespace tblis
{
namespace internal
{

static
void add_full(type_t type, const communicator& comm, const cntx_t* cntx,
              const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
              const dim_vector& idx_A_A,
              const dim_vector& idx_A_AB,
              const scalar&  beta, bool conj_B, const dpd_marray_view<char>& B,
              const dim_vector& idx_B_B,
              const dim_vector& idx_B_AB)
{
    auto [A2, len_A2, stride_A2] = block_to_full(type, comm, cntx, A);
    auto [B2, len_B2, stride_B2] = block_to_full(type, comm, cntx, B);

    auto len_A = stl_ext::select_from(len_A2, idx_A_A);
    auto len_B = stl_ext::select_from(len_B2, idx_B_B);
    auto len_AB = stl_ext::select_from(len_A2, idx_A_AB);
    auto stride_A_A = stl_ext::select_from(stride_A2, idx_A_A);
    auto stride_B_B = stl_ext::select_from(stride_B2, idx_B_B);
    auto stride_A_AB = stl_ext::select_from(stride_A2, idx_A_AB);
    auto stride_B_AB = stl_ext::select_from(stride_B2, idx_B_AB);

    add(type, comm, cntx, len_A, len_B, len_AB,
        alpha, conj_A, A2, stride_A_A, stride_A_AB,
         beta, conj_B, B2, stride_B_B, stride_B_AB);

    full_to_block(type, comm, cntx, B2, len_B2, stride_B2, B);

    if (comm.master())
    {
        delete[] A2;
        delete[] B2;
    }
}

static
void trace_block(type_t type, const communicator& comm, const cntx_t* cntx,
                 const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
                 const dim_vector& idx_A,
                 const dim_vector& idx_A_AB,
                 const scalar&  beta, bool conj_B, const dpd_marray_view<char>& B,
                 const dim_vector& idx_B_AB)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();
    const auto irrep_AB = B.irrep();
    const auto irrep_A = A.irrep()^irrep_AB;
    const auto ndim_A = A.dimension();
    const auto ndim_B = B.dimension();
    const int ndim_A_only = idx_A.size();
    const int ndim_AB = idx_A_AB.size();

    stride_type nblock_A = ipow(nirrep, ndim_A_only-1);
    stride_type nblock_AB = ipow(nirrep, ndim_AB-1);

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);

    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
    {
        assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                      irreps_A, idx_A_AB, irreps_B, idx_B_AB);

        if (is_block_empty(B, irreps_B)) continue;

        marray_view<char> local_B = B(irreps_B);

        auto len_AB = stl_ext::select_from(local_B.lengths(), idx_B_AB);
        auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

        auto local_beta = beta;
        auto local_conj_B = conj_B;

        for (stride_type block_A = 0;block_A < nblock_A;block_A++)
        {
            assign_irreps(ndim_A_only, irrep_A, nirrep, block_A,
                  irreps_A, idx_A);

            if (is_block_empty(A, irreps_A)) continue;

            marray_view<char> local_A = A(irreps_A);

            auto len_A_only = stl_ext::select_from(local_A.lengths(), idx_A);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
            auto stride_A_A = stl_ext::select_from(local_A.strides(), idx_A);

            add(type, comm, cntx, len_A_only, {}, len_AB,
                     alpha,       conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_A, stride_A_AB,
                local_beta, local_conj_B, B.data() + (local_B.data()-B.data())*ts,         {}, stride_B_AB);

            local_beta = 1.0;
            local_conj_B = false;
        }

        if (local_beta.is_zero())
        {
            set(type, comm, cntx, local_B.lengths(),
                local_beta, B.data() + (local_B.data()-B.data())*ts, local_B.strides());
        }
        else if (!local_beta.is_one() || (local_beta.is_complex() && local_conj_B))
        {
            scale(type, comm, cntx, local_B.lengths(),
                  local_beta, local_conj_B, B.data() + (local_B.data()-B.data())*ts, local_B.strides());
        }
    }
}

static
void replicate_block(type_t type, const communicator& comm, const cntx_t* cntx,
                     const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
                     const dim_vector& idx_A_AB,
                     const scalar&  beta, bool conj_B, const dpd_marray_view<char>& B,
                     const dim_vector& idx_B,
                     const dim_vector& idx_B_AB)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();
    const auto irrep_AB = A.irrep();
    const auto irrep_B = B.irrep()^irrep_AB;
    const auto ndim_A = A.dimension();
    const auto ndim_B = B.dimension();
    const int ndim_B_only = idx_B.size();
    const int ndim_AB = idx_A_AB.size();

    stride_type nblock_B = ipow(nirrep, ndim_B_only-1);
    stride_type nblock_AB = ipow(nirrep, ndim_AB-1);

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);

    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
    {
        assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

        if (is_block_empty(A, irreps_A)) continue;

        marray_view<char> local_A = A(irreps_A);

        auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
        auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);

        for (stride_type block_B = 0;block_B < nblock_B;block_B++)
        {
            assign_irreps(ndim_B_only, irrep_B, nirrep, block_B,
                          irreps_B, idx_B);

            if (is_block_empty(B, irreps_B)) continue;

            marray_view<char> local_B = B(irreps_B);

            auto len_B_only = stl_ext::select_from(local_B.lengths(), idx_B);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);
            auto stride_B_B = stl_ext::select_from(local_B.strides(), idx_B);

            add(type, comm, cntx, {}, len_B_only, len_AB,
                alpha, conj_A, A.data() + (local_A.data()-A.data())*ts,         {}, stride_A_AB,
                 beta, conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_B, stride_B_AB);
        }
    }

    if (beta.is_one() && !(beta.is_complex() && conj_B)) return;

    for (auto irrep_AB : range(nirrep))
    {
        if (irrep_AB == A.irrep()) continue;

        const auto irrep_B = B.irrep()^irrep_AB;

        for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
        {
            assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                          irreps_B, idx_B_AB);

            for (stride_type block_B = 0;block_B < nblock_B;block_B++)
            {
                assign_irreps(ndim_B_only, irrep_B, nirrep, block_B,
                              irreps_B, idx_B);

                if (is_block_empty(B, irreps_B)) continue;

                marray_view<char> local_B = B(irreps_B);

                if (beta.is_zero())
                {
                    set(type, comm, cntx, local_B.lengths(),
                        beta, B.data() + (local_B.data()-B.data())*ts, local_B.strides());
                }
                else
                {
                    scale(type, comm, cntx, local_B.lengths(),
                          beta, conj_B, B.data() + (local_B.data()-B.data())*ts, local_B.strides());
                }
            }
        }
    }
}

static
void transpose_block(type_t type, const communicator& comm, const cntx_t* cntx,
                     const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
                     const dim_vector& idx_A_AB,
                     const scalar&  beta, bool conj_B, const dpd_marray_view<char>& B,
                     const dim_vector& idx_B_AB)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();
    const auto irrep_AB = A.irrep();
    const auto ndim_A = A.dimension();
    const auto ndim_B = B.dimension();
    const int ndim_AB = idx_A_AB.size();

    stride_type nblock_AB = ipow(nirrep, ndim_AB-1);

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);

    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
    {
        assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                      irreps_A, idx_A_AB, irreps_B, idx_B_AB);

        if (is_block_empty(A, irreps_A)) continue;

        marray_view<char> local_A = A(irreps_A);
        marray_view<char> local_B = B(irreps_B);

        auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
        auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

        add(type, comm, cntx, {}, {}, len_AB,
            alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, {}, stride_A_AB,
             beta, conj_B, B.data() + (local_B.data()-B.data())*ts, {}, stride_B_AB);
    }
}

void add(type_t type, const communicator& comm, const cntx_t* cntx,
         const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
         const dim_vector& idx_A,
         const dim_vector& idx_A_AB,
         const scalar&  beta, bool conj_B, const dpd_marray_view<char>& B,
         const dim_vector& idx_B,
         const dim_vector& idx_B_AB)
{
    if (dpd_impl == FULL)
    {
        add_full(type, comm, cntx,
                 alpha, conj_A, A, idx_A, idx_A_AB,
                  beta, conj_B, B, idx_B, idx_B_AB);
    }
    else if (!idx_A.empty())
    {
        trace_block(type, comm, cntx,
                    alpha, conj_A, A, idx_A, idx_A_AB,
                     beta, conj_B, B, idx_B_AB);
    }
    else if (!idx_B.empty())
    {
        replicate_block(type, comm, cntx,
                        alpha, conj_A, A, idx_A_AB,
                         beta, conj_B, B, idx_B, idx_B_AB);
    }
    else
    {
        transpose_block(type, comm, cntx,
                        alpha, conj_A, A, idx_A_AB,
                         beta, conj_B, B, idx_B_AB);
    }
}

}
}
