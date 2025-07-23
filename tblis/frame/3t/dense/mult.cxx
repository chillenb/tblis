#include "mult.hpp"

#include "tblis/frame/base/tensor.hpp"
#include "tblis/frame/base/block_scatter.hpp"

#include "tblis/frame/0/add.hpp"
#include "tblis/frame/0/mult.hpp"
#include "tblis/frame/1t/dense/add.hpp"
#include "tblis/frame/1t/dense/dot.hpp"
#include "tblis/frame/1t/dense/scale.hpp"
#include "tblis/frame/1t/dense/set.hpp"

#include "tblis/frame/1m/packm/packm_blk_bsmtc.hpp"
#include "tblis/frame/3m/gemm/gemm_ker_bsmtc.hpp"

#include "tblis/plugin/bli_plugin_tblis.h"

#include <numeric>

using gemm_vfp = void (*)(      trans_t transa, \
                                trans_t transb, \
                                dim_t   m, \
                                dim_t   n, \
                                dim_t   k, \
                          const void*   alpha, \
                          const void*   a, inc_t rs_a, inc_t cs_a, \
                          const void*   b, inc_t rs_b, inc_t cs_b, \
                          const void*   beta, \
                                void*   c, inc_t rs_c, inc_t cs_c);

GENARRAY_FPA(gemm_vfp, gemm);

namespace tblis
{
namespace internal
{

impl_t impl = BLIS_BASED;

static
void ger_blis(type_t type, const communicator& comm, const cntx_t* cntx,
              len_type m, len_type n,
              const scalar& alpha, bool conj_A, const char* A, stride_type inc_A,
                                   bool conj_B, const char* B, stride_type inc_B,
              const scalar&  beta, bool conj_C,       char* C, stride_type rs_C, stride_type cs_C)
{
    auto ts = type_size[type];
    scalar zero(0.0, type);
    scalar one(1.0, type);

    auto scal2v_ukr = reinterpret_cast<scal2v_ker_ft>(bli_cntx_get_ukr_dt((num_t)type, BLIS_SCAL2V_KER, cntx));
    auto axpbyv_ukr = reinterpret_cast<axpbyv_ker_ft>(bli_cntx_get_ukr_dt((num_t)type, BLIS_AXPBYV_KER, cntx));

    comm.distribute_over_threads(m, n,
    [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
    {
        auto A1 = A + m_min*inc_A*ts;
        auto B1 = B +                  n_min*inc_B*ts;
        auto C1 = C + m_min* rs_C*ts + n_min* cs_C*ts;

        if (rs_C <= cs_C)
        {
            scalar alpha_B(0.0, type);

            for (auto j : range(n_min, n_max))
            {
                add(type, alpha, conj_B, B1, zero, false, alpha_B.raw());

                if (conj_C && bli_dt_dom_is_complex((num_t)type))
                    scal2v_ukr(BLIS_CONJUGATE, m_max-m_min, one.raw(), C1, rs_C, C1, rs_C, cntx);

                axpbyv_ukr(conj_A ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                           m_max-m_min,
                           alpha_B.raw(), A1, inc_A,
                              beta.raw(), C1,  rs_C,
                           cntx);

                B1 += inc_B*ts;
                C1 +=  cs_C*ts;
            }
        }
        else
        {
            scalar alpha_A(0.0, type);

            for (auto i : range(m_min, m_max))
            {
                add(type, alpha, conj_A, A1, zero, false, alpha_A.raw());

                if (conj_C && bli_dt_dom_is_complex((num_t)type))
                    scal2v_ukr(BLIS_CONJUGATE, n_max-n_min, one.raw(), C1, cs_C, C1, cs_C, cntx);

                axpbyv_ukr(conj_B ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                           n_max-n_min,
                           alpha_A.raw(), B1, inc_B,
                              beta.raw(), C1,  cs_C,
                           cntx);

                A1 += inc_A*ts;
                C1 +=  rs_C*ts;
            }
        }
    });
}

static
void gemv_blis(type_t type, const communicator& comm, const cntx_t* cntx,
               len_type m, len_type n,
               const scalar& alpha, bool conj_A, const char* A, stride_type rs_A, stride_type cs_A,
                                    bool conj_B, const char* B, stride_type inc_B,
               const scalar&  beta, bool conj_C,       char* C, stride_type inc_C)
{
    auto ts = type_size[type];
    scalar zero(0.0, type);
    scalar one(1.0, type);

    auto AF = bli_cntx_get_blksz_def_dt((num_t)type, BLIS_AF, cntx);
    auto DF = bli_cntx_get_blksz_def_dt((num_t)type, BLIS_DF, cntx);

    auto setv_ukr   = reinterpret_cast<setv_ker_ft >(bli_cntx_get_ukr_dt((num_t)type, BLIS_SETV_KER, cntx));
    auto scal2v_ukr = reinterpret_cast<scal2v_ker_ft>(bli_cntx_get_ukr_dt((num_t)type, BLIS_SCAL2V_KER, cntx));
    auto dotxf_ukr  = reinterpret_cast<dotxf_ker_ft>(bli_cntx_get_ukr_dt((num_t)type, BLIS_DOTXF_KER, cntx));
    auto axpyf_ukr  = reinterpret_cast<axpyf_ker_ft>(bli_cntx_get_ukr_dt((num_t)type, BLIS_AXPYF_KER, cntx));

    comm.distribute_over_threads({m, DF},
    [&](len_type m_min, len_type m_max)
    {
        auto A1 = A + m_min* rs_A*ts;
        auto B1 = B;
        auto C1 = C + m_min*inc_C*ts;

        if (beta.is_zero())
            setv_ukr(BLIS_NO_CONJUGATE, m_max-m_min, zero.raw(), C1, inc_C, cntx);
        else if ((conj_C && bli_dt_dom_is_complex((num_t)type)) || !beta.is_one())
            scal2v_ukr(conj_C ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE, m_max-m_min, beta.raw(), C1, inc_C, C1, inc_C, cntx);

        if (rs_A <= cs_A)
        {
            for (auto j : range(0,n,AF))
            {
                axpyf_ukr(conj_A ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                          conj_B ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                          m_max-m_min, std::min(AF, n-j),
                          alpha.raw(), A1, rs_A, cs_A,
                                       B1, inc_B,
                                       C1, inc_C,
                          cntx);

                A1 += AF* cs_A*ts;
                B1 += AF*inc_B*ts;
            }
        }
        else
        {
            for (auto i : range(m_min,m_max,DF))
            {
                dotxf_ukr(conj_A ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                          conj_B ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                          n, std::min(DF, m_max-i),
                          alpha.raw(), A1, cs_A, rs_A,
                                       B1, inc_B,
                            one.raw(), C1, inc_C,
                          cntx);

                A1 += DF* rs_A*ts;
                C1 += DF*inc_C*ts;
            }
        }
    });
}

static
void gemm_blis(type_t type, const communicator& comm, const cntx_t* cntx,
               len_type m, len_type n, len_type k,
               const scalar& alpha, bool conj_A, const char* A, stride_type rs_A, stride_type cs_A,
                                    bool conj_B, const char* B, stride_type rs_B, stride_type cs_B,
               const scalar&  beta, bool conj_C,       char* C, stride_type rs_C, stride_type cs_C)
{
    obj_t ao, bo, co, alpo, beto;

    bli_obj_create_with_attached_buffer((num_t)type, m, k, (void*)A, rs_A, cs_A, &ao);
    bli_obj_create_with_attached_buffer((num_t)type, k, n, (void*)B, rs_B, cs_B, &bo);
    bli_obj_create_with_attached_buffer((num_t)type, m, n, (void*)C, rs_C, cs_C, &co);

    bli_obj_create_1x1_with_attached_buffer((num_t)type, (void*)alpha.raw(), &alpo);
    bli_obj_create_1x1_with_attached_buffer((num_t)type, (void*)beta.raw(), &beto);

    if (conj_A) bli_obj_toggle_conj(&ao);
    if (conj_B) bli_obj_toggle_conj(&bo);

    if (conj_C && !beta.is_zero() && bli_dt_dom_is_complex((num_t)type))
    {
        bli_obj_toggle_conj(&co);
        bli_scal2m(&BLIS_ONE, &co, &co);
        bli_obj_toggle_conj(&co);
    }

    if ( bli_l3_return_early_if_trivial( &alpo, &ao, &bo, &beto, &co ) == BLIS_SUCCESS )
        return;

    gemm_cntl_t cntl;
    bli_gemm_cntl_init
    (
      bli_dt_dom_is_complex((num_t)type) ? bli_ind_oper_find_avail(BLIS_GEMM, (num_t)type) : BLIS_NAT,
      BLIS_GEMM,
      &alpo,
      &ao,
      &bo,
      &beto,
      &co,
      cntx,
      &cntl
    );

    thread_blis(comm, &ao, &bo, &co, cntx, (cntl_t*)&cntl);
}

void gemm_bsmtc_blis(type_t type, const communicator& comm, const cntx_t* cntx,
                     std::span<const len_type> len_AC, bool pack_3d_AC,
                     std::span<const len_type> len_BC, bool pack_3d_BC,
                     std::span<const len_type> len_AB, bool pack_3d_AB,
                     const scalar& alpha, bool conj_A, const char* A, std::span<const stride_type> block_off_A_AC, std::span<const stride_type> block_off_A_AB, std::span<const stride_type> stride_A_AC, std::span<const stride_type> stride_A_AB,
                                          bool conj_B, const char* B, std::span<const stride_type> block_off_B_BC, std::span<const stride_type> block_off_B_AB, std::span<const stride_type> stride_B_BC, std::span<const stride_type> stride_B_AB,
                     const scalar& beta_, bool conj_C,       char* C, std::span<const stride_type> block_off_C_AC, std::span<const stride_type> block_off_C_BC, std::span<const stride_type> stride_C_AC, std::span<const stride_type> stride_C_BC)
{
    stride_type zero = 0;
    if (!block_off_A_AC.data()) block_off_A_AC = std::span(&zero, 1);
    if (!block_off_A_AB.data()) block_off_A_AB = std::span(&zero, 1);
    if (!block_off_B_BC.data()) block_off_B_BC = std::span(&zero, 1);
    if (!block_off_B_AB.data()) block_off_B_AB = std::span(&zero, 1);
    if (!block_off_C_AC.data()) block_off_C_AC = std::span(&zero, 1);
    if (!block_off_C_BC.data()) block_off_C_BC = std::span(&zero, 1);

    int ndim_AC = len_AC.size();
    int ndim_BC = len_BC.size();
    int ndim_AB = len_AB.size();
    len_type nblock_AC = block_off_A_AC.size();
    len_type nblock_BC = block_off_B_BC.size();
    len_type nblock_AB = block_off_A_AB.size();

    TBLIS_ASSERT(len_AC.size() == stride_A_AC.size());
    TBLIS_ASSERT(len_AC.size() == stride_C_AC.size());
    TBLIS_ASSERT(len_BC.size() == stride_B_BC.size());
    TBLIS_ASSERT(len_BC.size() == stride_C_BC.size());
    TBLIS_ASSERT(len_AB.size() == stride_A_AB.size());
    TBLIS_ASSERT(len_AB.size() == stride_B_AB.size());
    TBLIS_ASSERT(block_off_A_AC.size() == block_off_C_AC.size());
    TBLIS_ASSERT(block_off_B_BC.size() == block_off_C_BC.size());
    TBLIS_ASSERT(block_off_A_AB.size() == block_off_B_AB.size());

    obj_t ao, bo, co, alpo, beto;
    auto beta = beta_;

    auto m = std::reduce(len_AC.begin(), len_AC.end(), len_type{1}, std::multiplies<len_type>{}) * nblock_AC;
    auto n = std::reduce(len_BC.begin(), len_BC.end(), len_type{1}, std::multiplies<len_type>{}) * nblock_BC;
    auto k = std::reduce(len_AB.begin(), len_AB.end(), len_type{1}, std::multiplies<len_type>{}) * nblock_AB;

    auto at = (ndim_AB ? stride_A_AB[0] : 1) < (ndim_AC ? stride_A_AC[0] : 1);
    auto bt = (ndim_BC ? stride_B_BC[0] : 1) < (ndim_AB ? stride_B_AB[0] : 1);
    auto ct = (ndim_BC ? stride_C_BC[0] : 1) < (ndim_AC ? stride_C_AC[0] : 1);

    bli_obj_create_with_attached_buffer((num_t)type, m, k, (void*)A, at ? k : 1, at ? 1 : m, &ao);
    bli_obj_create_with_attached_buffer((num_t)type, k, n, (void*)B, bt ? n : 1, bt ? 1 : k, &bo);
    bli_obj_create_with_attached_buffer((num_t)type, m, n, (void*)C, ct ? n : 1, ct ? 1 : m, &co);

    bli_obj_create_1x1_with_attached_buffer((num_t)type, (void*)alpha.raw(), &alpo);
    bli_obj_create_1x1_with_attached_buffer((num_t)type, (void*)beta.raw(), &beto);

    if (conj_A) bli_obj_toggle_conj(&ao);
    if (conj_B) bli_obj_toggle_conj(&bo);

    if (conj_C && !beta.is_zero() && bli_dt_dom_is_complex((num_t)type))
    {
        auto ts = type_size[type];

        len_vector len_C(len_AC.begin(), len_AC.end());
        len_C.insert(len_C.end(), len_BC.begin(), len_BC.end());

        len_vector stride_C(stride_C_AC.begin(), stride_C_AC.end());
        stride_C.insert(stride_C.end(), stride_B_BC.begin(), stride_B_BC.end());

        for (auto i : range(nblock_AC))
        for (auto j : range(nblock_BC))
            scale(type, comm, cntx, len_C, beta, conj_C, C + block_off_C_AC[i]*ts + block_off_C_BC[j]*ts, stride_C);

        beta = 1.0;
        conj_C = false;
    }

    gemm_cntl_t cntl;
    auto trans = bli_gemm_cntl_init
    (
      bli_dt_dom_is_complex((num_t)type) ? bli_ind_oper_find_avail(BLIS_GEMM, (num_t)type) : BLIS_NAT,
      BLIS_GEMM,
      &alpo,
      &ao,
      &bo,
      &beto,
      &co,
      cntx,
      &cntl
    );

    bsmtc_params params_A, params_B, params_C;

    params_A.nblock = {nblock_AC, nblock_AB};
    params_A.block_off = {block_off_A_AC.data(), block_off_A_AB.data()};
    params_A.ndim = {ndim_AC, ndim_AB};
    params_A.len = {len_AC.data(), len_AB.data()};
    params_A.stride = {stride_A_AC.data(), stride_A_AB.data()};
    params_A.pack_3d = {pack_3d_AC, pack_3d_AB};

    params_B.nblock = {nblock_BC, nblock_AB};
    params_B.block_off = {block_off_B_BC.data(), block_off_B_AB.data()};
    params_B.ndim = {ndim_BC, ndim_AB};
    params_B.len = {len_BC.data(), len_AB.data()};
    params_B.stride = {stride_B_BC.data(), stride_B_AB.data()};
    params_B.pack_3d = {pack_3d_BC, pack_3d_AB};

    params_C.nblock = {nblock_AC, nblock_BC};
    params_C.block_off = {block_off_C_AC.data(), block_off_C_BC.data()};
    params_C.ndim = {ndim_AC, ndim_BC};
    params_C.len = {len_AC.data(), len_BC.data()};
    params_C.stride = {stride_C_AC.data(), stride_C_BC.data()};
    params_C.pack_3d = {pack_3d_AC, pack_3d_BC};

    bli_gemm_cntl_set_packa_var(packm_blk_bsmtc, &cntl);
    bli_gemm_cntl_set_packb_var(packm_blk_bsmtc, &cntl);
    bli_gemm_cntl_set_var(gemm_ker_bsmtc, &cntl);

    bli_gemm_cntl_set_packa_params(trans ? &params_B : &params_A, &cntl);
    bli_gemm_cntl_set_packb_params(trans ? &params_A : &params_B, &cntl);
    bli_gemm_cntl_set_params(&params_C, &cntl);

    if (trans)
    {
        auto swap = [](auto& a)
        {
            using std::swap;
            swap(a[0], a[1]);
        };

        swap(params_C.nblock);
        swap(params_C.block_off);
        swap(params_C.ndim);
        swap(params_C.len);
        swap(params_C.stride);
        swap(params_C.pack_3d);
    }

    thread_blis(comm, &ao, &bo, &co, cntx, (cntl_t*)&cntl);
}

static
void mult_blis(type_t type, const communicator& comm, const cntx_t* cntx,
               const len_vector& len_AB,
               const len_vector& len_AC,
               const len_vector& len_ABC,
               const scalar& alpha, bool conj_A, const char* A,
               const stride_vector& stride_A_AB_,
               const stride_vector& stride_A_AC_,
               const stride_vector& stride_A_ABC_,
                                    bool conj_B, const char* B,
               const stride_vector& stride_B_AB_,
               const stride_vector& stride_B_ABC_,
               const scalar&  beta, bool conj_C,       char* C,
               const stride_vector& stride_C_AC_,
               const stride_vector& stride_C_ABC_)
{
    const len_type ts = type_size[type];

    auto stride_A_AB = stride_A_AB_;
    auto stride_A_AC = stride_A_AC_;
    auto stride_A_ABC = stride_A_ABC_;
    auto stride_B_AB = stride_B_AB_;
    auto stride_B_ABC = stride_B_ABC_;
    auto stride_C_AC = stride_C_AC_;
    auto stride_C_ABC = stride_C_ABC_;

    auto reorder_AC = internal::sort_by_stride(stride_A_AC, stride_C_AC);
    auto reorder_AB = internal::sort_by_stride(stride_A_AB, stride_B_AB);
    auto reorder_ABC = internal::sort_by_stride(stride_C_ABC, stride_A_ABC, stride_B_ABC);

    auto m = len_AC[reorder_AC[0]];
    auto n = len_AB[reorder_AB[0]];
    auto rs_A = stride_A_AC[reorder_AC[0]];
    auto cs_A = stride_A_AB[reorder_AB[0]];
    auto inc_B = stride_B_AB[reorder_AB[0]];
    auto inc_C = stride_C_AC[reorder_AC[0]];

    reorder_AC.erase(reorder_AC.begin());
    reorder_AB.erase(reorder_AB.begin());

    for (auto& s : stride_A_AC) s *= ts;
    for (auto& s : stride_A_AB) s *= ts;
    for (auto& s : stride_B_AB) s *= ts;
    for (auto& s : stride_C_AC) s *= ts;
    for (auto& s : stride_A_ABC) s *= ts;
    for (auto& s : stride_B_ABC) s *= ts;
    for (auto& s : stride_C_ABC) s *= ts;

    len_type l = stl_ext::prod(len_ABC);
    len_type m2 = stl_ext::prod(len_AC)/m;
    len_type n2 = stl_ext::prod(len_AB)/n;

    if (comm.master()) flops += 2*m*m2*n*n2*l;

    unsigned nt_l, nt_m;
    std::tie(nt_l, nt_m) = partition_2x2(comm.num_threads(), l*m2, l*m2, m, m);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    subcomm.distribute_over_gangs(l*m2,
    [&](len_type l_min, len_type l_max)
    {
        viterator<2> iter_AB(stl_ext::permuted(len_AB, reorder_AB),
                             stl_ext::permuted(stride_A_AB, reorder_AB),
                             stl_ext::permuted(stride_B_AB, reorder_AB));

        viterator<3> iter_ABC(stl_ext::appended(stl_ext::permuted(len_ABC, reorder_ABC),
                                                stl_ext::permuted(len_AC, reorder_AC)),
                              stl_ext::appended(stl_ext::permuted(stride_A_ABC, reorder_ABC),
                                                stl_ext::permuted(stride_A_AC, reorder_AC)),
                              stl_ext::appended(stl_ext::permuted(stride_B_ABC, reorder_ABC),
                                                stride_vector(reorder_AC.size())),
                              stl_ext::appended(stl_ext::permuted(stride_C_ABC, reorder_ABC),
                                                stl_ext::permuted(stride_C_AC, reorder_AC)));

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_ABC.position(l_min, A1, B1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_ABC.next(A1, B1, C1);

            auto beta1 = beta;
            auto conj_C1 = conj_C;

            while (iter_AB.next(A1, B1))
            {
                gemv_blis(type, subcomm, cntx,
                          m, n,
                          alpha, conj_A,  A1, rs_A, cs_A,
                                 conj_B,  B1, inc_B,
                          beta1, conj_C1, C1, inc_C);

                subcomm.barrier();

                beta1 = 1.0;
                conj_C1 = false;
            }
        }
    });
}

static
void mult_blis(type_t type, const communicator& comm, const cntx_t* cntx,
               const len_vector& len_AC,
               const len_vector& len_BC,
               const len_vector& len_ABC,
               const scalar& alpha,
               bool conj_A, const char* A,
               const stride_vector& stride_A_AC_,
               const stride_vector& stride_A_ABC_,
               bool conj_B, const char* B,
               const stride_vector& stride_B_BC_,
               const stride_vector& stride_B_ABC_,
               const scalar& beta,
               bool conj_C,       char* C,
               const stride_vector& stride_C_AC_,
               const stride_vector& stride_C_BC_,
               const stride_vector& stride_C_ABC_)
{
    const len_type ts = type_size[type];

    auto stride_A_AC = stride_A_AC_;
    auto stride_A_ABC = stride_A_ABC_;
    auto stride_B_BC = stride_B_BC_;
    auto stride_B_ABC = stride_B_ABC_;
    auto stride_C_AC = stride_C_AC_;
    auto stride_C_BC = stride_C_BC_;
    auto stride_C_ABC = stride_C_ABC_;

    auto reorder_AC = internal::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = internal::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_ABC = internal::sort_by_stride(stride_C_ABC, stride_A_ABC, stride_B_ABC);

    auto m = len_AC[reorder_AC[0]];
    auto n = len_BC[reorder_BC[0]];
    auto rs_C = stride_C_AC[reorder_AC[0]];
    auto cs_C = stride_C_BC[reorder_BC[0]];
    auto inc_A = stride_A_AC[reorder_AC[0]];
    auto inc_B = stride_B_BC[reorder_BC[0]];

    reorder_AC.erase(reorder_AC.begin());
    reorder_BC.erase(reorder_BC.begin());

    for (auto& s : stride_A_AC) s *= ts;
    for (auto& s : stride_B_BC) s *= ts;
    for (auto& s : stride_C_AC) s *= ts;
    for (auto& s : stride_C_BC) s *= ts;
    for (auto& s : stride_A_ABC) s *= ts;
    for (auto& s : stride_B_ABC) s *= ts;
    for (auto& s : stride_C_ABC) s *= ts;

    len_type l = stl_ext::prod(len_ABC);
    len_type m2 = stl_ext::prod(len_AC)/m;
    len_type n2 = stl_ext::prod(len_BC)/n;

    if (comm.master()) flops += 2*m*m2*n*n2*l;

    unsigned nt_l, nt_m;
    std::tie(nt_l, nt_m) = partition_2x2(comm.num_threads(), l*m2*n2, l*m2*n2, m*n, m*n);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    subcomm.distribute_over_gangs(l*m2*n2,
    [&](len_type l_min, len_type l_max)
    {
        viterator<3> iter_ABC(stl_ext::appended(stl_ext::permuted(len_ABC, reorder_ABC),
                                                stl_ext::permuted(len_AC, reorder_AC),
                                                stl_ext::permuted(len_BC, reorder_BC)),
                              stl_ext::appended(stl_ext::permuted(stride_A_ABC, reorder_ABC),
                                                stl_ext::permuted(stride_A_AC, reorder_AC),
                                                stride_vector(reorder_BC.size())),
                              stl_ext::appended(stl_ext::permuted(stride_B_ABC, reorder_ABC),
                                                stride_vector(reorder_AC.size()),
                                                stl_ext::permuted(stride_B_BC, reorder_BC)),
                              stl_ext::appended(stl_ext::permuted(stride_C_ABC, reorder_ABC),
                                                stl_ext::permuted(stride_C_AC, reorder_AC),
                                                stl_ext::permuted(stride_C_BC, reorder_BC)));

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_ABC.position(l_min, A1, B1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_ABC.next(A1, B1, C1);

            ger_blis(type, subcomm, cntx,
                     m, n,
                     alpha, conj_A, A1, inc_A,
                            conj_B, B1, inc_B,
                      beta, conj_C, C1, rs_C, cs_C);
        }
    });
}

static
void mult_blis(type_t type, const communicator& comm, const cntx_t* cntx,
               const len_vector& len_AB,
               const len_vector& len_AC,
               const len_vector& len_BC,
               const len_vector& len_ABC,
               const scalar& alpha,
               bool conj_A, const char* A,
               const stride_vector& stride_A_AB,
               const stride_vector& stride_A_AC,
               const stride_vector& stride_A_ABC,
               bool conj_B, const char* B,
               const stride_vector& stride_B_AB,
               const stride_vector& stride_B_BC,
               const stride_vector& stride_B_ABC,
               const scalar& beta,
               bool conj_C,       char* C,
               const stride_vector& stride_C_AC,
               const stride_vector& stride_C_BC,
               const stride_vector& stride_C_ABC)
{
    const len_type ts = type_size[type];

    auto reorder_AC = internal::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = internal::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_AB = internal::sort_by_stride(stride_A_AB, stride_B_AB);
    auto reorder_ABC = internal::sort_by_stride(stride_C_ABC, stride_A_ABC, stride_B_ABC);

    auto unit_A_AC = unit_dim(stride_A_AC, reorder_AC);
    auto unit_C_AC = unit_dim(stride_C_AC, reorder_AC);
    auto unit_B_BC = unit_dim(stride_B_BC, reorder_BC);
    auto unit_C_BC = unit_dim(stride_C_BC, reorder_BC);
    auto unit_A_AB = unit_dim(stride_A_AB, reorder_AB);
    auto unit_B_AB = unit_dim(stride_B_AB, reorder_AB);

    TBLIS_ASSERT(unit_C_AC == 0 || unit_C_AC == (int)len_AC.size());
    TBLIS_ASSERT(unit_C_BC == 0 || unit_C_BC == (int)len_BC.size());
    TBLIS_ASSERT(unit_A_AB == 0 || unit_B_AB == 0 ||
                 (unit_A_AB == (int)len_AB.size() &&
                  unit_B_AB == (int)len_AB.size()));

    //bool pack_M_3d = unit_A_AC > 0 && unit_A_AC < (int)len_AC.size();
    //bool pack_N_3d = unit_B_BC > 0 && unit_B_BC < (int)len_BC.size();
    //bool pack_K_3d = (unit_A_AB > 0 && unit_A_AB < (int)len_AB.size()) ||
    //                 (unit_B_AB > 0 && unit_B_AB < (int)len_AB.size());

    bool pack_M_3d = false;
    bool pack_N_3d = false;
    bool pack_K_3d = false;

    if (pack_M_3d)
        std::rotate(reorder_AC.begin()+1, reorder_AC.begin()+unit_A_AC, reorder_AC.end());

    if (pack_N_3d)
        std::rotate(reorder_BC.begin()+1, reorder_BC.begin()+unit_B_BC, reorder_BC.end());

    if (pack_K_3d)
        std::rotate(reorder_AB.begin()+1, reorder_AB.begin()+std::max(unit_A_AB, unit_B_AB), reorder_AB.end());

    scalar one(1.0, type);

    len_type m = stl_ext::prod(len_AC);
    len_type n = stl_ext::prod(len_BC);
    len_type k = stl_ext::prod(len_AB);
    len_type l = stl_ext::prod(len_ABC);

    if (comm.master()) flops += 2*m*n*k*l;

    unsigned nt_l, nt_mn;
    std::tie(nt_l, nt_mn) =
        partition_2x2(comm.num_threads(), l, l, m*n, m*n);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    auto len_AB_r = stl_ext::permuted(len_AB, reorder_AB);
    auto len_AC_r = stl_ext::permuted(len_AC, reorder_AC);
    auto len_BC_r = stl_ext::permuted(len_BC, reorder_BC);
    auto stride_A_AB_r = stl_ext::permuted(stride_A_AB, reorder_AB);
    auto stride_B_AB_r = stl_ext::permuted(stride_B_AB, reorder_AB);
    auto stride_A_AC_r = stl_ext::permuted(stride_A_AC, reorder_AC);
    auto stride_C_AC_r = stl_ext::permuted(stride_C_AC, reorder_AC);
    auto stride_B_BC_r = stl_ext::permuted(stride_B_BC, reorder_BC);
    auto stride_C_BC_r = stl_ext::permuted(stride_C_BC, reorder_BC);
    auto stride_A_ABC_r = stl_ext::permuted(stride_A_ABC, reorder_ABC);
    auto stride_B_ABC_r = stl_ext::permuted(stride_B_ABC, reorder_ABC);
    auto stride_C_ABC_r = stl_ext::permuted(stride_C_ABC, reorder_ABC);

    subcomm.distribute_over_gangs(l,
    [&](len_type l_min, len_type l_max)
    {
        viterator<3> iter_ABC(stl_ext::permuted(len_ABC, reorder_ABC),
                              stride_A_ABC_r, stride_B_ABC_r, stride_C_ABC_r);

        stride_type A1 = 0;
        stride_type B1 = 0;
        stride_type C1 = 0;

        iter_ABC.position(l_min, A1, B1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_ABC.next(A1, B1, C1);

            auto empty = make_span<stride_type>();
            gemm_bsmtc_blis(type, subcomm, cntx,
                            make_span(len_AC_r), pack_M_3d,
                            make_span(len_BC_r), pack_N_3d,
                            make_span(len_AB_r), pack_K_3d,
                            alpha, conj_A, A + A1*ts, empty, empty, make_span(stride_A_AC_r), make_span(stride_A_AB_r),
                                   conj_B, B + B1*ts, empty, empty, make_span(stride_B_BC_r), make_span(stride_B_AB_r),
                             beta, conj_C, C + C1*ts, empty, empty, make_span(stride_C_AC_r), make_span(stride_C_BC_r));
        }
    });
}

static
void mult_blas(type_t type, const communicator& comm, const cntx_t* cntx,
               const len_vector& len_AB_,
               const len_vector& len_AC_,
               const len_vector& len_BC_,
               const len_vector& len_ABC_,
               const scalar& alpha, bool conj_A, const char* A,
               const stride_vector& stride_A_AB_,
               const stride_vector& stride_A_AC_,
               const stride_vector& stride_A_ABC_,
                                    bool conj_B, const char* B,
               const stride_vector& stride_B_AB_,
               const stride_vector& stride_B_BC_,
               const stride_vector& stride_B_ABC_,
               const scalar&  beta, bool conj_C,       char* C,
               const stride_vector& stride_C_AC_,
               const stride_vector& stride_C_BC_,
               const stride_vector& stride_C_ABC_)
{
    auto ts = type_size[type];

    auto len_AB = len_AB_; len_AB.push_back(1);
    auto len_AC = len_AC_; len_AC.push_back(1);
    auto len_BC = len_BC_; len_BC.push_back(1);
    auto len_ABC = len_ABC_; len_ABC.push_back(1);
    auto stride_A_AB = stride_A_AB_; stride_A_AB.push_back(1);
    auto stride_B_AB = stride_B_AB_; stride_B_AB.push_back(1);
    auto stride_A_AC = stride_A_AC_; stride_A_AC.push_back(1);
    auto stride_C_AC = stride_C_AC_; stride_C_AC.push_back(1);
    auto stride_B_BC = stride_B_BC_; stride_B_BC.push_back(1);
    auto stride_C_BC = stride_C_BC_; stride_C_BC.push_back(1);
    auto stride_A_ABC = stride_A_ABC_; stride_A_ABC.push_back(1);
    auto stride_B_ABC = stride_B_ABC_; stride_B_ABC.push_back(1);
    auto stride_C_ABC = stride_C_ABC_; stride_C_ABC.push_back(1);

    auto stride_A = MArray::detail::strides(len_AC+len_AB, MArray::COLUMN_MAJOR);
    auto stride_B = MArray::detail::strides(len_BC+len_AB, MArray::COLUMN_MAJOR);
    auto stride_C = MArray::detail::strides(len_AC+len_BC, MArray::COLUMN_MAJOR);

    auto m = stl_ext::prod(len_AC);
    auto n = stl_ext::prod(len_BC);
    auto k = stl_ext::prod(len_AB);

    char *a = nullptr;
    char *b = nullptr;
    char *c = nullptr;
    if (comm.master())
    {
        a = new char[m * k * type_size[type]];
        b = new char[n * k * type_size[type]];
        c = new char[m * n * type_size[type]];
    }

    comm.broadcast(
    [&](auto a, auto b, auto c)
    {
        scalar one(1.0, type);
        scalar zero(0.0, type);

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

        while (it.next(A1, B1, C1))
        {
            add(type, comm, cntx, {}, {}, len_AC+len_AB,
                 one, conj_A, A + (A1-A)*ts, {}, stride_A_AC+stride_A_AB,
                zero,  false,             a, {}, stride_A);

            add(type, comm, cntx, {}, {}, len_BC+len_AB,
                 one, conj_B, B + (B1-B)*ts, {}, stride_B_BC+stride_B_AB,
                zero,  false,             b, {}, stride_B);

            gemm_blis(type, comm, cntx,
                      m, n, k,
                      alpha, false, a, 1, m,
                             false, b, n, 1,
                       zero, false, c, 1, m);

            add(type, comm, cntx, {}, {}, len_AC+len_BC,
                 one,  false,             c, {}, stride_C,
                beta, conj_C, C + (C1-C)*ts, {}, stride_C_AC+stride_C_BC);

            comm.barrier();
        }
    },
    a, b, c);

    if (comm.master())
    {
        delete[] a;
        delete[] b;
        delete[] c;
    }
}

static
void mult_ref(type_t type, const communicator& comm, const cntx_t* cntx,
              const len_vector& len_AB,
              const len_vector& len_AC,
              const len_vector& len_BC,
              const len_vector& len_ABC,
              const scalar& alpha,
              bool conj_A, const char* A,
              const stride_vector& stride_A_AB_,
              const stride_vector& stride_A_AC_,
              const stride_vector& stride_A_ABC_,
              bool conj_B, const char* B,
              const stride_vector& stride_B_AB_,
              const stride_vector& stride_B_BC_,
              const stride_vector& stride_B_ABC_,
              const scalar& beta,
              bool conj_C,       char* C,
              const stride_vector& stride_C_AC_,
              const stride_vector& stride_C_BC_,
              const stride_vector& stride_C_ABC_)
{
    const len_type ts = type_size[type];

    len_type n = stl_ext::prod(len_ABC);

    auto stride_A_AB = stride_A_AB_;
    auto stride_B_AB = stride_B_AB_;
    auto stride_A_AC = stride_A_AC_;
    auto stride_C_AC = stride_C_AC_;
    auto stride_B_BC = stride_B_BC_;
    auto stride_C_BC = stride_C_BC_;
    auto stride_A_ABC = stride_A_ABC_;
    auto stride_B_ABC = stride_B_ABC_;
    auto stride_C_ABC = stride_C_ABC_;

    for (auto& s : stride_A_AC) s *= ts;
    for (auto& s : stride_B_BC) s *= ts;
    for (auto& s : stride_C_AC) s *= ts;
    for (auto& s : stride_C_BC) s *= ts;
    for (auto& s : stride_A_ABC) s *= ts;
    for (auto& s : stride_B_ABC) s *= ts;
    for (auto& s : stride_C_ABC) s *= ts;

    comm.distribute_over_threads(n,
    [&](len_type n_min, len_type n_max)
    {
        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
        viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
        viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
        iter_ABC.position(n_min, A1, B1, C1);

        for (len_type i = n_min;i < n_max;i++)
        {
            iter_ABC.next(A1, B1, C1);

            while (iter_AC.next(A1, C1))
            {
                while (iter_BC.next(B1, C1))
                {
                    scalar sum(0, type);

                    dot(type, single, cntx, len_AB,
                        conj_A, A1, stride_A_AB,
                        conj_B, B1, stride_B_AB, sum.raw());

                    add(type, alpha, false, sum.raw(), beta, conj_C, C1);
                }
            }
        }
    });
}

static
void mult_vec(type_t type, const communicator& comm, const cntx_t* cntx,
              const len_vector& len_ABC,
              const scalar& alpha, bool conj_A, const char* A,
              const stride_vector& stride_A_ABC,
                                   bool conj_B, const char* B,
              const stride_vector& stride_B_ABC,
              const scalar&  beta, bool conj_C,       char* C,
              const stride_vector& stride_C_ABC)
{
    bool empty = len_ABC.size() == 0;

    const len_type ts = type_size[type];

    len_type n0 = (empty ? 1 : len_ABC[0]);
    len_vector len1(len_ABC.begin() + !empty, len_ABC.end());
    len_type n1 = stl_ext::prod(len1);

    stride_type stride_A0 = (empty ? 1 : stride_A_ABC[0]);
    stride_type stride_B0 = (empty ? 1 : stride_B_ABC[0]);
    stride_type stride_C0 = (empty ? 1 : stride_C_ABC[0]);
    len_vector stride_A1, stride_B1, stride_C1;
    for (auto i : range(1,stride_A_ABC.size())) stride_A1.push_back(stride_A_ABC[i]*ts);
    for (auto i : range(1,stride_B_ABC.size())) stride_B1.push_back(stride_B_ABC[i]*ts);
    for (auto i : range(1,stride_C_ABC.size())) stride_C1.push_back(stride_C_ABC[i]*ts);

    auto mult_ukr = reinterpret_cast<mult_ft>(bli_cntx_get_ukr_dt((num_t)type, MULT_KER, cntx));

    comm.distribute_over_threads(n0, n1,
    [&](len_type n0_min, len_type n0_max, len_type n1_min, len_type n1_max)
    {
        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        viterator<3> iter_ABC(len1, stride_A1, stride_B1, stride_C1);
        iter_ABC.position(n1_min, A1, B1, C1);
        A1 += n0_min*stride_A0*ts;
        B1 += n0_min*stride_B0*ts;
        C1 += n0_min*stride_C0*ts;

        for (len_type i = n1_min;i < n1_max;i++)
        {
            iter_ABC.next(A1, B1, C1);

            mult_ukr(n0_max-n0_min,
                     alpha.raw(), conj_A, A1, stride_A0,
                             conj_B, B1, stride_B0,
                      beta.raw(), conj_C, C1, stride_C0);
        }
    });
}

void mult(type_t type, const communicator& comm, const cntx_t* cntx,
          const len_vector& len_AB,
          const len_vector& len_AC,
          const len_vector& len_BC,
          const len_vector& len_ABC,
          const scalar& alpha, bool conj_A, const char* A,
          const stride_vector& stride_A_AB,
          const stride_vector& stride_A_AC,
          const stride_vector& stride_A_ABC,
                               bool conj_B, const char* B,
          const stride_vector& stride_B_AB,
          const stride_vector& stride_B_BC,
          const stride_vector& stride_B_ABC,
          const scalar&  beta, bool conj_C,       char* C,
          const stride_vector& stride_C_AC,
          const stride_vector& stride_C_BC,
          const stride_vector& stride_C_ABC)
{
    bli_init();

    const len_type ts = type_size[type];
    auto n_AB = stl_ext::prod(len_AB);
    auto n_AC = stl_ext::prod(len_AC);
    auto n_BC = stl_ext::prod(len_BC);
    auto n_ABC = stl_ext::prod(len_ABC);

    if (n_AC == 0 || n_BC == 0 || n_ABC == 0) return;

    if (n_AB == 0)
    {
        if (beta.is_zero())
        {
            set(type, comm, cntx, len_AC+len_BC+len_ABC, beta, C,
                stride_C_AC+stride_C_BC+stride_C_ABC);
        }
        else if (!beta.is_one() || (bli_dt_dom_is_complex((num_t)type) && conj_C))
        {
            scale(type, comm, cntx, len_AC+len_BC+len_ABC, beta, conj_C, C,
                  stride_C_AC+stride_C_BC+stride_C_ABC);
        }

        return;
    }

    if (impl == REFERENCE)
    {
        mult_ref(type, comm, cntx,
                 len_AB, len_AC, len_BC, len_ABC,
                 alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                        conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                  beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        comm.barrier();
        return;
    }
    else if (impl == BLAS_BASED)
    {
        mult_blas(type, comm, cntx,
                  len_AB, len_AC, len_BC, len_ABC,
                  alpha, conj_A, A,
                         stride_A_AB, stride_A_AC, stride_A_ABC,
                         conj_B, B,
                         stride_B_AB, stride_B_BC, stride_B_ABC,
                   beta, conj_C, C,
                         stride_C_AC, stride_C_BC, stride_C_ABC);
        comm.barrier();
        return;
    }

    enum
    {
        HAS_NONE = 0x0,
        HAS_AB   = 0x1,
        HAS_AC   = 0x2,
        HAS_BC   = 0x4,
        HAS_ABC  = 0x8
    };

    int groups = (n_AB  == 1 ? 0 : HAS_AB ) +
                 (n_AC  == 1 ? 0 : HAS_AC ) +
                 (n_BC  == 1 ? 0 : HAS_BC ) +
                 (n_ABC == 1 ? 0 : HAS_ABC);

    scalar zero(0, type);
    scalar sum(0, type);
    auto stride_A_ABC_ts = stride_A_ABC; for (auto& s : stride_A_ABC_ts) s *= ts;
    auto stride_B_ABC_ts = stride_B_ABC; for (auto& s : stride_B_ABC_ts) s *= ts;
    auto stride_C_ABC_ts = stride_C_ABC; for (auto& s : stride_C_ABC_ts) s *= ts;
    viterator<3> iter_ABC(len_ABC, stride_A_ABC_ts, stride_B_ABC_ts, stride_C_ABC_ts);

    switch (groups)
    {
        case HAS_NONE:
        {
            if (comm.master())
                mult(type, alpha, conj_A, A, conj_B, B, beta, conj_C, C);
        }
        break;
        case HAS_ABC:
        {
            mult_vec(type, comm, cntx, len_ABC,
                     alpha, conj_A, A, stride_A_ABC,
                            conj_B, B, stride_B_ABC,
                      beta, conj_C, C, stride_C_ABC);
        }
        break;
        case HAS_AB:
        case HAS_AB+HAS_ABC:
        {
            while (iter_ABC.next(A, B, C))
            {
                dot(type, comm, cntx, len_AB, conj_A, A, stride_A_AB,
                                              conj_B, B, stride_B_AB, sum.raw());

                if (comm.master())
                    add(type, alpha, false, sum.raw(), beta, conj_C, C);
            }
        }
        break;
        case HAS_AC:
        case HAS_AC+HAS_ABC:
        {
            while (iter_ABC.next(A, B, C))
            {
                add(type, alpha, conj_B, B, zero, false, sum.raw());

                add(type, comm, cntx, {}, {}, len_AC,
                     sum, conj_A, A, {}, stride_A_AC,
                    beta, conj_C, C, {}, stride_C_AC);
            }
        }
        break;
        case HAS_BC:
        case HAS_BC+HAS_ABC:
        {
            while (iter_ABC.next(A, B, C))
            {
                add(type, alpha, conj_A, A, zero, false, sum.raw());

                add(type, comm, cntx, {}, {}, len_BC,
                     sum, conj_B, B, {}, stride_B_BC,
                    beta, conj_C, C, {}, stride_C_BC);
            }
        }
        break;
        case HAS_AC+HAS_BC:
        case HAS_AC+HAS_BC+HAS_ABC:
        {
            mult_blis(type, comm, cntx, len_AC, len_BC, len_ABC,
                      alpha, conj_A, A, stride_A_AC, stride_A_ABC,
                             conj_B, B, stride_B_BC, stride_B_ABC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC:
        case HAS_AB+HAS_AC+HAS_ABC:
        {
            mult_blis(type, comm, cntx,
                      len_AB, len_AC, len_ABC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                             conj_B, B, stride_B_AB, stride_B_ABC,
                       beta, conj_C, C, stride_C_AC, stride_C_ABC);
        }
        break;
        case HAS_AB+HAS_BC:
        case HAS_AB+HAS_BC+HAS_ABC:
        {
            mult_blis(type, comm, cntx,
                      len_AB, len_BC, len_ABC,
                      alpha, conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                             conj_A, A, stride_A_AB, stride_A_ABC,
                       beta, conj_C, C, stride_C_BC, stride_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_BC:
        case HAS_AB+HAS_AC+HAS_BC+HAS_ABC:
        {
            mult_blis(type, comm, cntx,
                      len_AB, len_AC, len_BC, len_ABC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                             conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        }
        break;
    }

    comm.barrier();
}

}
}
