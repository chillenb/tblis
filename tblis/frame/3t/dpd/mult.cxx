#include "mult.hpp"

#include "tblis/frame/0/add.hpp"
#include "tblis/frame/0/mult.hpp"
#include "tblis/frame/1t/dense/scale.hpp"
#include "tblis/frame/1t/dense/set.hpp"
#include "tblis/frame/1t/dpd/util.hpp"
#include "tblis/frame/1t/dpd/add.hpp"
#include "tblis/frame/1t/dpd/dot.hpp"
#include "tblis/frame/1t/dpd/scale.hpp"
#include "tblis/frame/1t/dpd/set.hpp"
#include "tblis/frame/3t/dense/mult.hpp"

#include "tblis/frame/base/tensor.hpp"
#include "tblis/frame/base/dpd_block_scatter.hpp"

#include "tblis/frame/1m/packm/packm_blk_dpd.hpp"
#include "tblis/frame/3m/gemm/gemm_ker_dpd.hpp"

#include "tblis/plugin/bli_plugin_tblis.h"

namespace tblis
{
namespace internal
{

dpd_impl_t dpd_impl = BLIS;

void gemm_dpd_blis(type_t type, const communicator& comm, const cntx_t* cntx,
                   int nirrep, int irrep_AC, int irrep_BC, int irrep_AB,
                   bool pack_3d_AC, bool pack_3d_BC, bool pack_3d_AB,
                   const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
                   const dim_vector idx_A_AC, int unit_A_AC,
                   const dim_vector idx_A_AB, int unit_A_AB,
                   const dim_vector& extra_dim_A,
                   const irrep_vector& extra_irreps_A,
                   const len_vector& extra_idx_A,
                                        bool conj_B, const dpd_marray_view<char>& B,
                   const dim_vector& idx_B_BC, int unit_B_BC,
                   const dim_vector& idx_B_AB, int unit_B_AB,
                   const dim_vector& extra_dim_B,
                   const irrep_vector& extra_irreps_B,
                   const len_vector& extra_idx_B,
                                                     const dpd_marray_view<char>& C,
                   const dim_vector& idx_C_AC, int unit_C_AC,
                   const dim_vector& idx_C_BC, int unit_C_BC,
                   const dim_vector& extra_dim_C,
                   const irrep_vector& extra_irreps_C,
                   const len_vector& extra_idx_C)
{
    auto ts = type_size[type];
    obj_t ao, bo, co, alpo;

    dpd_params params_A(A, idx_A_AC, idx_A_AB, irrep_AB, extra_dim_A, extra_irreps_A, extra_idx_A, pack_3d_AC, pack_3d_AB);
    dpd_params params_B(B, idx_B_BC, idx_B_AB, irrep_AB, extra_dim_B, extra_irreps_B, extra_idx_B, pack_3d_BC, pack_3d_AB);
    dpd_params params_C(C, idx_C_AC, idx_C_BC, irrep_BC, extra_dim_C, extra_irreps_C, extra_idx_C, pack_3d_AC, pack_3d_BC);

    auto m = stl_ext::sum(params_C.patch_size[0]);
    auto n = stl_ext::sum(params_C.patch_size[1]);
    auto k = stl_ext::sum(params_A.patch_size[1]);

    stride_type rs_A = 1, cs_A = m+k; if (unit_A_AC > unit_A_AB) std::swap(rs_A, cs_A);
    stride_type rs_B = 1, cs_B = n+k; if (unit_B_AB > unit_B_BC) std::swap(rs_B, cs_B);
    stride_type rs_C = 1, cs_C = m+n; if (unit_C_AC > unit_C_BC) std::swap(rs_C, cs_C);

    bli_obj_create_with_attached_buffer((num_t)type, m, k, (void*)A.data(), rs_A, cs_A, &ao);
    bli_obj_create_with_attached_buffer((num_t)type, k, n, (void*)B.data(), rs_B, cs_B, &bo);
    bli_obj_create_with_attached_buffer((num_t)type, m, n, (void*)C.data(), rs_C, cs_C, &co);

    bli_obj_create_1x1_with_attached_buffer((num_t)type, (void*)&alpha, &alpo);

    if (conj_A) bli_obj_toggle_conj(&ao);
    if (conj_B) bli_obj_toggle_conj(&bo);

    if (m == 0 || n == 0 || k == 0) return;

    gemm_cntl_t cntl;
    auto trans = bli_gemm_cntl_init
    (
      bli_dt_dom_is_complex((num_t)type) ? bli_ind_oper_find_avail(BLIS_GEMM, (num_t)type) : BLIS_NAT,
      BLIS_GEMM,
      &alpo,
      &ao,
      &bo,
      &BLIS_ONE,
      &co,
      cntx,
      &cntl
    );

    bli_gemm_cntl_set_packa_var(packm_blk_dpd, &cntl);
    bli_gemm_cntl_set_packb_var(packm_blk_dpd, &cntl);
    bli_gemm_cntl_set_var(gemm_ker_dpd, &cntl);

    bli_gemm_cntl_set_packa_params(trans ? &params_B : &params_A, &cntl);
    bli_gemm_cntl_set_packb_params(trans ? &params_A : &params_B, &cntl);
    bli_gemm_cntl_set_params(&params_C, &cntl);

    if (trans) params_C.transpose();

    thread_blis(comm, &ao, &bo, &co, cntx, (cntl_t*)&cntl);
}

static
void mult_full(type_t type, const communicator& comm, const cntx_t* cntx,
               const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
               const dim_vector& idx_A_AB,
               const dim_vector& idx_A_AC,
               const dim_vector& idx_A_ABC,
                                    bool conj_B, const dpd_marray_view<char>& B,
               const dim_vector& idx_B_AB,
               const dim_vector& idx_B_BC,
               const dim_vector& idx_B_ABC,
                                                 const dpd_marray_view<char>& C,
               const dim_vector& idx_C_AC,
               const dim_vector& idx_C_BC,
               const dim_vector& idx_C_ABC)
{
    scalar one(1.0, type);

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

    mult(type, comm, cntx, len_AB, len_AC, len_BC, len_ABC,
         alpha, conj_A, A2, stride_A_AB, stride_A_AC, stride_A_ABC,
                conj_B, B2, stride_B_AB, stride_B_BC, stride_B_ABC,
            one, false, C2, stride_C_AC, stride_C_BC, stride_C_ABC);

    full_to_block(type, comm, cntx, C2, len_C2, stride_C2, C);

    if (comm.master())
    {
        delete[] A2;
        delete[] B2;
        delete[] C2;
    }
}

static
void mult_blis(type_t type, const communicator& comm, const cntx_t* cntx,
               const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
               dim_vector idx_A_AB,
               dim_vector idx_A_AC,
               dim_vector idx_A_ABC,
                                    bool conj_B, const dpd_marray_view<char>& B,
               dim_vector idx_B_AB,
               dim_vector idx_B_ABC,
                                                 const dpd_marray_view<char>& C,
               dim_vector idx_C_AC,
               dim_vector idx_C_ABC)
{
    const len_type ts = type_size[type];
    scalar one(1.0, type);

    const auto nirrep = A.num_irreps();

    const auto irrep_ABC = A.irrep()^B.irrep()^C.irrep();
    const auto irrep_AB = A.irrep()^C.irrep();
    const auto irrep_AC = A.irrep()^B.irrep();

    irrep_iterator irrep_it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());
    irrep_iterator irrep_it_AC(irrep_AC, nirrep, idx_A_AC.size());
    irrep_iterator irrep_it_AB(irrep_AB, nirrep, idx_A_AB.size());

    irrep_vector irreps_A(A.dimension());
    irrep_vector irreps_B(B.dimension());
    irrep_vector irreps_C(C.dimension());

    while (irrep_it_ABC.next())
    while (irrep_it_AC.next())
    {
        for (auto i : range(idx_A_ABC.size()))
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = irrep_it_ABC.irrep(i);
        }

        for (auto i : range(idx_A_AC.size()))
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = irrep_it_AC.irrep(i);
        }

        marray_view<char> local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);

        while (irrep_it_AB.next())
        {
            for (auto i : range(idx_A_AB.size()))
            {
                irreps_A[idx_A_AB[i]] =
                irreps_B[idx_B_AB[i]] = irrep_it_AB.irrep(i);
            }

            marray_view<char> local_A = A(irreps_A);
            marray_view<char> local_B = B(irreps_B);

            auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
            auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
            auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
            auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

            mult(type, comm, cntx, len_AB, len_AC, {}, len_ABC,
                 alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_AB, stride_A_AC, stride_A_ABC,
                        conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_AB, {}, stride_B_ABC,
                   one,  false, C.data() + (local_C.data()-C.data())*ts, stride_C_AC, {}, stride_C_ABC);
        }
    }
}

static
void mult_blis(type_t type, const communicator& comm, const cntx_t* cntx,
               const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
               dim_vector idx_A_AC,
               dim_vector idx_A_ABC,
                                    bool conj_B, const dpd_marray_view<char>& B,
               dim_vector idx_B_BC,
               dim_vector idx_B_ABC,
                                                 const dpd_marray_view<char>& C,
               dim_vector idx_C_AC,
               dim_vector idx_C_BC,
               dim_vector idx_C_ABC)
{
    const len_type ts = type_size[type];
    scalar one(1.0, type);

    const auto nirrep = A.num_irreps();
    const auto irrep_ABC = A.irrep()^B.irrep()^C.irrep();
    const auto irrep_AC = A.irrep()^irrep_ABC;
    const auto irrep_BC = B.irrep()^irrep_ABC;

    irrep_iterator irrep_it_ABC(irrep_ABC, nirrep, idx_C_ABC.size());
    irrep_iterator irrep_it_AC(irrep_AC, nirrep, idx_C_AC.size());
    irrep_iterator irrep_it_BC(irrep_BC, nirrep, idx_C_BC.size());

    irrep_vector irreps_A(A.dimension());
    irrep_vector irreps_B(B.dimension());
    irrep_vector irreps_C(C.dimension());

    while (irrep_it_ABC.next())
    while (irrep_it_AC.next())
    while (irrep_it_BC.next())
    {
        for (auto i : range(idx_A_ABC.size()))
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = irrep_it_ABC.irrep(i);
        }

        for (auto i : range(idx_A_AC.size()))
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = irrep_it_AC.irrep(i);
        }

        for (auto i : range(idx_B_BC.size()))
        {
            irreps_B[idx_B_BC[i]] =
            irreps_C[idx_C_BC[i]] = irrep_it_BC.irrep(i);
        }

        marray_view<char> local_A = A(irreps_A);
        marray_view<char> local_B = B(irreps_B);
        marray_view<char> local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
        auto len_BC = stl_ext::select_from(local_C.lengths(), idx_C_BC);
        auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
        auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);
        auto stride_B_BC = stl_ext::select_from(local_B.strides(), idx_B_BC);
        auto stride_C_BC = stl_ext::select_from(local_C.strides(), idx_C_BC);

        mult(type, comm, cntx, {}, len_AC, len_BC, len_ABC,
             alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, {}, stride_A_AC, stride_A_ABC,
                    conj_B, B.data() + (local_B.data()-B.data())*ts, {}, stride_B_BC, stride_B_ABC,
               one,  false, C.data() + (local_C.data()-C.data())*ts, stride_C_AC, stride_C_BC, stride_C_ABC);
    }
}

static
void mult_blis(type_t type, const communicator& comm, const cntx_t* cntx,
               const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
               dim_vector idx_A_AB,
               dim_vector idx_A_AC,
               dim_vector idx_A_ABC,
                                    bool conj_B, const dpd_marray_view<char>& B,
               dim_vector idx_B_AB,
               dim_vector idx_B_BC,
               dim_vector idx_B_ABC,
                                                 const dpd_marray_view<char>& C,
               dim_vector idx_C_AC,
               dim_vector idx_C_BC,
               dim_vector idx_C_ABC)
{
    const auto nirrep = A.num_irreps();
    scalar one(1.0, type);

    std::array<len_vector,3> len;
    std::array<stride_vector,3> stride;
    dense_total_lengths_and_strides(len, stride, A, idx_A_AB, B, idx_B_AB, C, idx_C_AC);

    auto perm_AC = internal::sort_by_stride(stl_ext::select_from(stride[2], idx_C_AC),
                                          stl_ext::select_from(stride[0], idx_A_AC));
    auto perm_BC = internal::sort_by_stride(stl_ext::select_from(stride[2], idx_C_BC),
                                          stl_ext::select_from(stride[1], idx_B_BC));
    auto perm_AB = internal::sort_by_stride(stl_ext::select_from(stride[0], idx_A_AB),
                                          stl_ext::select_from(stride[1], idx_B_AB));
    auto perm_ABC = internal::sort_by_stride(stl_ext::select_from(stride[2], idx_A_ABC),
                                           stl_ext::select_from(stride[0], idx_B_ABC),
                                           stl_ext::select_from(stride[1], idx_C_ABC));

    stl_ext::permute(idx_A_AC, perm_AC);
    stl_ext::permute(idx_A_AB, perm_AB);
    stl_ext::permute(idx_B_AB, perm_AB);
    stl_ext::permute(idx_B_BC, perm_BC);
    stl_ext::permute(idx_C_AC, perm_AC);
    stl_ext::permute(idx_C_BC, perm_BC);
    stl_ext::permute(idx_A_ABC, perm_ABC);
    stl_ext::permute(idx_B_ABC, perm_ABC);
    stl_ext::permute(idx_C_ABC, perm_ABC);

    auto unit_A_AC = unit_dim(stride[0], idx_A_AC);
    auto unit_C_AC = unit_dim(stride[2], idx_C_AC);
    auto unit_B_BC = unit_dim(stride[1], idx_B_BC);
    auto unit_C_BC = unit_dim(stride[2], idx_C_BC);
    auto unit_A_AB = unit_dim(stride[0], idx_A_AB);
    auto unit_B_AB = unit_dim(stride[1], idx_B_AB);

    TBLIS_ASSERT(unit_C_AC == 0 || unit_C_AC == (int)perm_AC.size());
    TBLIS_ASSERT(unit_C_BC == 0 || unit_C_BC == (int)perm_BC.size());
    TBLIS_ASSERT(unit_A_AB == 0 || unit_B_AB == 0 ||
                 (unit_A_AB == (int)perm_AB.size() &&
                  unit_B_AB == (int)perm_AB.size()));

    //bool pack_M_3d = unit_A_AC > 0 && unit_A_AC < (int)perm_AC.size();
    //bool pack_N_3d = unit_B_BC > 0 && unit_B_BC < (int)perm_BC.size();
    //bool pack_K_3d = (unit_A_AB > 0 && unit_A_AB < (int)perm_AB.size()) ||
    //                 (unit_B_AB > 0 && unit_B_AB < (int)perm_AB.size());

    bool pack_M_3d = false;
    bool pack_N_3d = false;
    bool pack_K_3d = false;

    if (pack_M_3d)
    {
        std::rotate(idx_A_AC.begin()+1, idx_A_AC.begin()+unit_A_AC, idx_A_AC.end());
        std::rotate(idx_C_AC.begin()+1, idx_C_AC.begin()+unit_A_AC, idx_C_AC.end());
    }

    if (pack_N_3d)
    {
        std::rotate(idx_B_BC.begin()+1, idx_B_BC.begin()+unit_B_BC, idx_B_BC.end());
        std::rotate(idx_C_BC.begin()+1, idx_C_BC.begin()+unit_B_BC, idx_C_BC.end());
    }

    if (pack_K_3d)
    {
        auto unit_AB = std::max(unit_A_AB, unit_B_AB);
        std::rotate(idx_A_AB.begin()+1, idx_A_AB.begin()+unit_AB, idx_A_AB.end());
        std::rotate(idx_B_AB.begin()+1, idx_B_AB.begin()+unit_AB, idx_B_AB.end());
    }

    irrep_vector irreps_ABC(idx_A_ABC.size());
    len_vector len_ABC(idx_A_ABC.size());

    for (auto irrep_ABC : range(nirrep))
    for (auto irrep_AB : range(nirrep))
    {
        auto irrep_AC = A.irrep()^irrep_ABC^irrep_AB;
        auto irrep_BC = C.irrep()^irrep_ABC^irrep_AC;

        irrep_iterator irrep_it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());

        while (irrep_it_ABC.next())
        {
            for (auto i : range(idx_A_ABC.size()))
            {
                irreps_ABC[i] = irrep_it_ABC.irrep(i);
                len_ABC[i] = A.length(idx_A_ABC[i], irreps_ABC[i]);
            }

            viterator<0> it_ABC(len_ABC);

            while (it_ABC.next())
            {
                gemm_dpd_blis(type, comm, cntx,
                              nirrep, irrep_AC, irrep_BC, irrep_AB,
                              pack_M_3d, pack_N_3d, pack_K_3d,
                              alpha, conj_A, A, idx_A_AC, unit_A_AC, idx_A_AB, unit_A_AB,
                              idx_A_ABC, irreps_ABC, it_ABC.position(),
                                     conj_B, B, idx_B_BC, unit_B_BC, idx_B_AB, unit_B_AB,
                              idx_B_ABC, irreps_ABC, it_ABC.position(),
                                             C, idx_C_AC, unit_C_AC, idx_C_BC, unit_C_BC,
                              idx_C_ABC, irreps_ABC, it_ABC.position());
            }
        }
    }
}

static
void mult_block(type_t type, const communicator& comm, const cntx_t* cntx,
                const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
                dim_vector idx_A_AB,
                dim_vector idx_A_AC,
                dim_vector idx_A_ABC,
                                     bool conj_B, const dpd_marray_view<char>& B,
                dim_vector idx_B_AB,
                dim_vector idx_B_BC,
                dim_vector idx_B_ABC,
                                                  const dpd_marray_view<char>& C,
                dim_vector idx_C_AC,
                dim_vector idx_C_BC,
                dim_vector idx_C_ABC)
{
    const len_type ts = type_size[type];
    scalar one(1.0, type);

    const auto nirrep = A.num_irreps();
    const auto ndim_A = A.dimension();
    const auto ndim_B = B.dimension();
    const auto ndim_C = C.dimension();
    const int ndim_AC = idx_C_AC.size();
    const int ndim_BC = idx_C_BC.size();
    const int ndim_AB = idx_A_AB.size();
    const int ndim_ABC = idx_A_ABC.size();

    std::array<len_vector,3> len;
    std::array<stride_vector,3> stride;
    dense_total_lengths_and_strides(len, stride, A, idx_A_AB, B, idx_B_AB,
                                    C, idx_C_AC);

    auto perm_AC = internal::sort_by_stride(stl_ext::select_from(stride[2], idx_C_AC),
                                          stl_ext::select_from(stride[0], idx_A_AC));
    auto perm_BC = internal::sort_by_stride(stl_ext::select_from(stride[2], idx_C_BC),
                                          stl_ext::select_from(stride[1], idx_B_BC));
    auto perm_AB = internal::sort_by_stride(stl_ext::select_from(stride[0], idx_A_AB),
                                          stl_ext::select_from(stride[1], idx_B_AB));

    stl_ext::permute(idx_A_AC, perm_AC);
    stl_ext::permute(idx_A_AB, perm_AB);
    stl_ext::permute(idx_B_AB, perm_AB);
    stl_ext::permute(idx_B_BC, perm_BC);
    stl_ext::permute(idx_C_AC, perm_AC);
    stl_ext::permute(idx_C_BC, perm_BC);

    stride_type nblock_AB = ipow(nirrep, ndim_AB-1);
    stride_type nblock_AC = ipow(nirrep, ndim_AC-1);
    stride_type nblock_BC = ipow(nirrep, ndim_BC-1);
    stride_type nblock_ABC = ipow(nirrep, ndim_ABC-1);

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);
    irrep_vector irreps_C(ndim_C);

    for (auto irrep_ABC : range(nirrep))
    {
        if (ndim_ABC == 0 && irrep_ABC != 0) continue;

        for (auto irrep_AB : range(nirrep))
        {
            auto irrep_AC = A.irrep()^irrep_ABC^irrep_AB;
            auto irrep_BC = C.irrep()^irrep_ABC^irrep_AC;

            if (ndim_AC == 0 && irrep_AC != 0) continue;
            if (ndim_BC == 0 && irrep_BC != 0) continue;

            for (stride_type block_ABC = 0;block_ABC < nblock_ABC;block_ABC++)
            {
                assign_irreps(ndim_ABC, irrep_ABC, nirrep, block_ABC,
                              irreps_A, idx_A_ABC, irreps_B, idx_B_ABC, irreps_C, idx_C_ABC);

                for (stride_type block_AC = 0;block_AC < nblock_AC;block_AC++)
                {
                    assign_irreps(ndim_AC, irrep_AC, nirrep, block_AC,
                                  irreps_A, idx_A_AC, irreps_C, idx_C_AC);

                    for (stride_type block_BC = 0;block_BC < nblock_BC;block_BC++)
                    {
                        assign_irreps(ndim_BC, irrep_BC, nirrep, block_BC,
                                      irreps_B, idx_B_BC, irreps_C, idx_C_BC);

                        if (is_block_empty(C, irreps_C)) continue;

                        marray_view<char> local_C = C(irreps_C);

                        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
                        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
                        auto len_BC = stl_ext::select_from(local_C.lengths(), idx_C_BC);
                        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
                        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);
                        auto stride_C_BC = stl_ext::select_from(local_C.strides(), idx_C_BC);

                        if ((ndim_AB != 0 || irrep_AB == 0) &&
                            irrep_ABC == (A.irrep()^B.irrep()^C.irrep()))
                        {
                            for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                            {
                                assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                                              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

                                if (is_block_empty(A, irreps_A)) continue;

                                marray_view<char> local_A = A(irreps_A);
                                marray_view<char> local_B = B(irreps_B);

                                auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
                                auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
                                auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
                                auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);
                                auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
                                auto stride_B_BC = stl_ext::select_from(local_B.strides(), idx_B_BC);

                                mult(type, comm, cntx, len_AB, len_AC, len_BC, len_ABC,
                                     alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_AB, stride_A_AC, stride_A_ABC,
                                            conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_AB, stride_B_BC, stride_B_ABC,
                                       one,  false, C.data() + (local_C.data()-C.data())*ts, stride_C_AC, stride_C_BC, stride_C_ABC);
                            }
                        }
                    }
                }
            }
        }
    }
}

static
void mult_vec(type_t type, const communicator& comm, const cntx_t* cntx,
              const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
              dim_vector idx_A_ABC,
                                   bool conj_B, const dpd_marray_view<char>& B,
              dim_vector idx_B_ABC,
                                                const dpd_marray_view<char>& C,
              dim_vector idx_C_ABC)
{
    if (A.irrep() != B.irrep() || A.irrep() != C.irrep())
        return;

    const len_type ts = type_size[type];
    scalar one(1.0, type);

    const auto nirrep = A.num_irreps();

    irrep_vector irreps_A(idx_A_ABC.size());
    irrep_vector irreps_B(idx_A_ABC.size());
    irrep_vector irreps_C(idx_A_ABC.size());

    const auto irrep_ABC = C.irrep();

    irrep_iterator it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());

    while (it_ABC.next())
    {
        for (auto i : range(1,idx_A_ABC.size()))
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = it_ABC.irrep(i);
        }

        if (is_block_empty(C, irreps_C)) continue;

        marray_view<char> local_A = A(irreps_A);
        marray_view<char> local_B = B(irreps_B);
        marray_view<char> local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);

        mult(type, comm, cntx, {}, {}, {}, len_ABC,
             alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, {}, {}, stride_A_ABC,
                    conj_B, B.data() + (local_B.data()-B.data())*ts, {}, {}, stride_B_ABC,
               one,  false, C.data() + (local_C.data()-C.data())*ts, {}, {}, stride_C_ABC);
    }
}

static
void mult_vec(type_t type, const communicator& comm, const cntx_t* cntx,
              const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
              dim_vector idx_A_AB,
              dim_vector idx_A_ABC,
                                   bool conj_B, const dpd_marray_view<char>& B,
              dim_vector idx_B_AB,
              dim_vector idx_B_ABC,
                                                const dpd_marray_view<char>& C,
              dim_vector idx_C_ABC)
{
    if (A.irrep() != B.irrep())
        return;

    const len_type ts = type_size[type];
    scalar one(1.0, type);

    const auto nirrep = A.num_irreps();

    irrep_vector irreps_A(idx_A_ABC.size());
    irrep_vector irreps_B(idx_A_ABC.size());
    irrep_vector irreps_C(idx_A_ABC.size());

    auto irrep_ABC = C.irrep();
    auto irrep_AB = A.irrep()^irrep_ABC;

    irrep_iterator it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());
    irrep_iterator it_AB(irrep_AB, nirrep, idx_A_ABC.size());

    while (it_ABC.next())
    {
        for (auto i : range(1,idx_A_ABC.size()))
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = it_ABC.irrep(i);
        }

        if (is_block_empty(C, irreps_C)) continue;

        marray_view<char> local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);

        while (it_AB.next())
        {
            for (auto i : range(idx_A_AB.size()))
            {
                irreps_A[idx_A_AB[i]] =
                irreps_B[idx_B_AB[i]] = it_AB.irrep(i);
            }

            marray_view<char> local_A = A(irreps_A);
            marray_view<char> local_B = B(irreps_B);

            auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
            auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
            auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

            mult(type, comm, cntx, len_AB, {}, {}, len_ABC,
                 alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_AB, {}, stride_A_ABC,
                        conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_AB, {}, stride_B_ABC,
                   one,  false, C.data() + (local_C.data()-C.data())*ts, {}, {}, stride_C_ABC);
        }
    }
}

static
void mult_vec(type_t type, const communicator& comm, const cntx_t* cntx,
              const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
              dim_vector idx_A_AC,
              dim_vector idx_A_ABC,
                                   bool conj_B, const dpd_marray_view<char>& B,
              dim_vector idx_B_ABC,
                                                const dpd_marray_view<char>& C,
              dim_vector idx_C_AC,
              dim_vector idx_C_ABC)
{
    if (A.irrep() != C.irrep())
        return;

    const len_type ts = type_size[type];
    scalar one(1.0, type);

    const auto nirrep = A.num_irreps();

    irrep_vector irreps_A(idx_A_ABC.size());
    irrep_vector irreps_B(idx_A_ABC.size());
    irrep_vector irreps_C(idx_A_ABC.size());

    auto irrep_ABC = B.irrep();
    auto irrep_AC = A.irrep()^irrep_ABC;

    irrep_iterator it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());
    irrep_iterator it_AC(irrep_AC, nirrep, idx_A_AC.size());

    while (it_ABC.next())
    while (it_AC.next())
    {
        for (auto i : range(1,idx_A_ABC.size()))
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = it_ABC.irrep(i);
        }

        for (auto i : range(1,idx_A_AC.size()))
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = it_AC.irrep(i);
        }

        if (is_block_empty(C, irreps_C)) continue;

        marray_view<char> local_A = A(irreps_A);
        marray_view<char> local_B = B(irreps_B);
        marray_view<char> local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto len_AC = stl_ext::select_from(local_A.lengths(), idx_A_AC);
        auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
        auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
        auto stride_C_AC = stl_ext::select_from(local_B.strides(), idx_C_AC);

        mult(type, comm, cntx, {}, len_AC, {}, len_ABC,
             alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, {}, stride_A_AC, stride_A_ABC,
                    conj_B, B.data() + (local_B.data()-B.data())*ts, {}, {}, stride_B_ABC,
               one,  false, C.data() + (local_C.data()-C.data())*ts, stride_C_AC, {}, stride_C_ABC);
    }
}

void mult(type_t type, const communicator& comm, const cntx_t* cntx,
          const scalar& alpha,
          bool conj_A, const dpd_marray_view<char>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
          bool conj_B, const dpd_marray_view<char>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          const scalar&  beta,
          bool conj_C, const dpd_marray_view<char>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC)
{
    bli_init();

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
        mult_full(type, comm, cntx,
                  alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                         conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                                 C, idx_C_AC, idx_C_BC, idx_C_ABC);

        comm.barrier();
        return;
    }
    else if (dpd_impl == BLOCKED)
    {
        mult_block(type, comm, cntx,
                   alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                          conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                                  C, idx_C_AC, idx_C_BC, idx_C_ABC);

        comm.barrier();
        return;
    }

    enum
    {
        HAS_NONE = 0x0,
        HAS_AB   = 0x1,
        HAS_AC   = 0x2,
        HAS_BC   = 0x4
    };

    int groups = (idx_A_AB.size()  == 0 ? 0 : HAS_AB ) +
                 (idx_A_AC.size()  == 0 ? 0 : HAS_AC ) +
                 (idx_B_BC.size()  == 0 ? 0 : HAS_BC );

    switch (groups)
    {
        case HAS_NONE:
        {
            mult_vec(type, comm, cntx,
                     alpha, conj_A, A, idx_A_ABC,
                            conj_B, B, idx_B_ABC,
                                    C, idx_C_ABC);
        }
        break;
        case HAS_AB:
        {
            mult_vec(type, comm, cntx,
                     alpha, conj_A, A, idx_A_AB, idx_A_ABC,
                            conj_B, B, idx_B_AB, idx_B_ABC,
                                    C, idx_C_ABC);
        }
        break;
        case HAS_AC:
        {
            mult_vec(type, comm, cntx,
                     alpha, conj_A, A, idx_A_AC, idx_A_ABC,
                            conj_B, B, idx_B_ABC,
                                    C, idx_C_AC, idx_C_ABC);
        }
        break;
        case HAS_BC:
        {
            mult_vec(type, comm, cntx,
                     alpha, conj_B, B, idx_B_BC, idx_B_ABC,
                            conj_A, A, idx_A_ABC,
                                    C, idx_C_BC, idx_C_ABC);
        }
        break;
        case HAS_AC+HAS_BC:
        {
            mult_blis(type, comm, cntx,
                      alpha, conj_A, A, idx_A_AC, idx_A_ABC,
                             conj_B, B, idx_B_BC, idx_B_ABC,
                                     C, idx_C_AC, idx_C_BC, idx_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC:
        {
            mult_blis(type, comm, cntx,
                      alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                             conj_B, B, idx_B_AB, idx_B_ABC,
                                     C, idx_C_AC, idx_C_ABC);
        }
        break;
        case HAS_AB+HAS_BC:
        {
            mult_blis(type, comm, cntx,
                      alpha, conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                             conj_A, A, idx_A_AB, idx_A_ABC,
                                     C, idx_C_BC, idx_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_BC:
        {
            mult_blis(type, comm, cntx,
                      alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                             conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                                     C, idx_C_AC, idx_C_BC, idx_C_ABC);
        }
        break;
    }

    comm.barrier();
}

}
}
