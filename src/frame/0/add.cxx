#include "add.hpp"

namespace tblis
{
namespace internal
{

void add(type_t type, const scalar& alpha, bool conj_A, const char* A,
                      const scalar&  beta, bool conj_B,       char* B)
{
    if (beta.is_zero())
    {
        switch (type)
        {
            case TYPE_FLOAT:
                *reinterpret_cast<float*>(B) =
                    alpha.data.s*(*reinterpret_cast<const float*>(A));
                break;
            case TYPE_DOUBLE:
                *reinterpret_cast<double*>(B) =
                    alpha.data.d*(*reinterpret_cast<const double*>(A));
                break;
            case TYPE_SCOMPLEX:
                *reinterpret_cast<scomplex*>(B) =
                    alpha.data.c*conj(conj_A, *reinterpret_cast<const scomplex*>(A));
                break;
            case TYPE_DCOMPLEX:
                *reinterpret_cast<dcomplex*>(B) =
                    alpha.data.z*conj(conj_A, *reinterpret_cast<const dcomplex*>(A));
                break;
        }
    }
    else
    {
        switch (type)
        {
            case TYPE_FLOAT:
                *reinterpret_cast<float*>(B) =
                    alpha.data.s*(*reinterpret_cast<const float*>(A)) +
                    beta.data.s*(*reinterpret_cast<float*>(B));
                break;
            case TYPE_DOUBLE:
                *reinterpret_cast<double*>(B) =
                    alpha.data.d*(*reinterpret_cast<const double*>(A)) +
                    beta.data.d*(*reinterpret_cast<double*>(B));
                break;
            case TYPE_SCOMPLEX:
                *reinterpret_cast<scomplex*>(B) =
                    alpha.data.c*conj(conj_A, *reinterpret_cast<const scomplex*>(A)) +
                    beta.data.c*conj(conj_B, *reinterpret_cast<scomplex*>(B));
                break;
            case TYPE_DCOMPLEX:
                *reinterpret_cast<dcomplex*>(B) =
                    alpha.data.z*conj(conj_A, *reinterpret_cast<const dcomplex*>(A)) +
                    beta.data.z*conj(conj_B, *reinterpret_cast<dcomplex*>(B));
                break;
        }
    }
}

}
}
