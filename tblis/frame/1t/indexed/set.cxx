#include "set.hpp"
#include "tblis/frame/1t/dense/set.hpp"

#include "tblis/frame/base/tensor.hpp"

namespace tblis
{
namespace internal
{

void set(type_t type, const communicator& comm, const cntx_t* cntx,
         const scalar& alpha, const indexed_marray_view<char>& A, const dim_vector&)
{
    for (len_type i = 0;i < A.num_indices();i++)
    {
        scalar alpha_fac = alpha;

        switch (type)
        {
            case TYPE_FLOAT:    alpha_fac.data.s *= reinterpret_cast<const indexed_marray_view<   float>&>(A).factor(i); break;
            case TYPE_DOUBLE:   alpha_fac.data.d *= reinterpret_cast<const indexed_marray_view<  double>&>(A).factor(i); break;
            case TYPE_SCOMPLEX: alpha_fac.data.c *= reinterpret_cast<const indexed_marray_view<scomplex>&>(A).factor(i); break;
            case TYPE_DCOMPLEX: alpha_fac.data.z *= reinterpret_cast<const indexed_marray_view<dcomplex>&>(A).factor(i); break;
        }

        set(type, comm, cntx, A.dense_lengths(), alpha_fac, A.data(i),
            A.dense_strides());
    }
}

}
}
