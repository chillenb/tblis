#include "block_scatter.hpp"
#include "dpd_block_scatter.hpp"

#if defined(TBLIS_HAVE_GCC_BITSET_BUILTINS)

template <typename T>
int popcount(T x)
{
    if constexpr (sizeof(T) == sizeof(unsigned long long))
    {
        return __builtin_popcountll(static_cast<unsigned long long>(x));
    }
    else if constexpr (sizeof(T) == sizeof(unsigned long))
    {
        return __builtin_popcountl(static_cast<unsigned long>(x));
    }
    else
    {
        return __builtin_popcount(static_cast<unsigned>(x));
    }
}

#elif defined(TBLIS_HAVE_CXX20_BITSET)

#include <bit>

template <typename T>
int popcount(T x)
{
    return std::popcount(static_cast<std::make_unsigned_t<T>>(x));
}

#else

template <typename T>
int popcount(T x_)
{
    auto x = static_cast<std::make_unsigned_t<T>>(x_);
    auto i = 0;
    auto c = 0;
    for (auto i = 0, b = 1;i < sizeof(x)*CHAR_BIT;i++, b <<= 1)
        if (x & b) c++;
    return c;
}

#endif

namespace tblis
{

dpd_params::dpd_params(const dpd_marray_view<char>& other,
                       const dim_vector& row_inds,
                       const dim_vector& col_inds,
                       int col_irrep,
                       const dim_vector& extra_inds,
                       const irrep_vector& extra_irreps,
                       const len_vector& extra_idx,
                       bool pack_m_3d, bool pack_n_3d)
: tensor(&other),
  dims{row_inds, col_inds},
  extra_dims(extra_inds),
  extra_irreps(extra_irreps),
  extra_idx(extra_idx),
  irrep{col_irrep^other.irrep(), col_irrep},
  pack_3d{pack_m_3d, pack_n_3d}
{
    for (auto i : extra_irreps)
        irrep[0] ^= i;

    TBLIS_ASSERT((int)(dims[0].size() + dims[1].size() + extra_dims.size()) == other.dimension());
    TBLIS_ASSERT(extra_dims.size() == extra_irreps.size());
    TBLIS_ASSERT(extra_dims.size() == extra_idx.size());

    const auto nirrep = other.num_irreps();

    for (auto& i : dims[0])
    for (auto& j : dims[1])
    {
        (void)i; (void)j;
        TBLIS_ASSERT(i != j);
    }

    for (auto dim : {0,1})
    {
        for (auto& i : dims[dim])
        for (auto& j : extra_dims)
        {
            (void)i; (void)j;
            TBLIS_ASSERT(i != j);
        }

        if (dims[dim].empty())
        {
            if (irrep[dim] == 0)
            {
                patch_size[dim].push_back(1);
                patch_idx[dim].push_back(0);
            }
        }
        else
        {
            MArray::irrep_iterator it(irrep[dim], nirrep, dims[dim].size());
            patch_size[dim].reserve(it.nblock());
            patch_idx[dim].reserve(it.nblock());
            for (int idx = 0;it.next();idx++)
            {
                stride_type size = 1;
                for (auto i : range(dims[dim].size()))
                    size *= other.length(dims[dim][i], it.irrep(i));

                if (size == 0) continue;

                patch_size[dim].push_back(size);
                patch_idx[dim].push_back(idx);
            }
        }
    }
}

void dpd_params::swap(dpd_params& other)
{
    using std::swap;
    swap(tensor, other.tensor);
    swap(dims, other.dims);
    swap(extra_dims, other.extra_dims);
    swap(extra_irreps, other.extra_irreps);
    swap(extra_idx, other.extra_idx);
    swap(irrep, other.irrep);
    swap(patch_size, other.patch_size);
    swap(patch_idx, other.patch_idx);
    swap(pack_3d, other.pack_3d);
}

void dpd_params::transpose()
{
    using std::swap;
    swap(dims[0], dims[1]);
    swap(irrep[0], irrep[1]);
    swap(patch_size[0], patch_size[1]);
    swap(patch_idx[0], patch_idx[1]);
    swap(pack_3d[0], pack_3d[1]);
}

char* fill_block_scatter(len_type ts, bool fill, const dpd_params& params,
                         len_type MR, len_type NR,
                         len_type m_patch, len_type m_off_patch, len_type m_patch_size,
                         len_type n_patch, len_type n_off_patch, len_type n_patch_size,
                         stride_type* rscat, stride_type* cscat,
                         stride_type* rbs, stride_type* cbs)
{
    const auto nirrep = params.tensor->num_irreps();
    const int irrep_mask = nirrep - 1;
    const int irrep_bits = popcount(irrep_mask);

    irrep_vector irreps(params.tensor->dimension());

    for (auto i : range(params.extra_dims.size()))
        irreps[params.extra_dims[i]] = params.extra_irreps[i];

    for (auto dim : {0,1})
    {
        auto& dims = params.dims[dim];

        if (dims.empty()) continue;

        auto idx = params.patch_idx[dim][dim == 0 ? m_patch : n_patch];
        TBLIS_ASSERT(idx < (1u << irrep_bits*(std::max<int>(dims.size(),1)-1)));
        irreps[dims[0]] = params.irrep[dim];
        for (auto i : range(1,dims.size()))
        {
            irreps[dims[0]] ^=
                irreps[dims[i]] = idx & irrep_mask;
            idx >>= irrep_bits;
        }
    }

    auto& A = *params.tensor;
    marray_view<char> A2 = A(irreps);

    auto len_m = stl_ext::select_from(A2.lengths(), params.dims[0]);
    auto len_n = stl_ext::select_from(A2.lengths(), params.dims[1]);
    auto stride_m = stl_ext::select_from(A2.strides(), params.dims[0]);
    auto stride_n = stl_ext::select_from(A2.strides(), params.dims[1]);

    auto p_a = A.data() + (A2.data()-A.data())*ts;

    for (auto i : range(params.extra_dims.size()))
        p_a += A2.stride(params.extra_dims[i]) * params.extra_idx[i]*ts;

    if (fill)
    {
        len_type zero = 0;

        fill_block_scatter(ts, 1, &zero,
                           len_m.size(), len_m.data(), stride_m.data(), MR,
                           m_off_patch, m_patch_size,
                           rscat, rbs,
                           params.pack_3d[0]);

        fill_block_scatter(ts, 1, &zero,
                           len_n.size(), len_n.data(), stride_n.data(), NR,
                           n_off_patch, n_patch_size,
                           cscat, cbs,
                           params.pack_3d[1]);
    }

    return p_a;
}

}
