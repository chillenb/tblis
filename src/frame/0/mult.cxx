#include "mult.hpp"

namespace tblis
{
namespace internal
{

void mult(type_t type, const scalar& alpha, bool conj_A, const char* A,
                                            bool conj_B, const char* B,
                       const scalar&  beta, bool conj_C,       char* C)
{
    if (beta.is_zero())
    {
        switch (type)
        {
            case TYPE_FLOAT:
                *reinterpret_cast<float*>(C) =
                    alpha.data.s*(*reinterpret_cast<const float*>(A)) *
                                 (*reinterpret_cast<const float*>(B));
                break;
            case TYPE_DOUBLE:
                *reinterpret_cast<double*>(C) =
                    alpha.data.d*(*reinterpret_cast<const double*>(A)) *
                                 (*reinterpret_cast<const double*>(B));
                break;
            case TYPE_SCOMPLEX:
                *reinterpret_cast<scomplex*>(C) =
                    alpha.data.c*conj(conj_A, *reinterpret_cast<const scomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<const scomplex*>(B));
                break;
            case TYPE_DCOMPLEX:
                *reinterpret_cast<dcomplex*>(C) =
                    alpha.data.z*conj(conj_A, *reinterpret_cast<const dcomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<const dcomplex*>(B));
                break;
        }
    }
    else
    {
        switch (type)
        {
            case TYPE_FLOAT:
                *reinterpret_cast<float*>(C) =
                    alpha.data.s*(*reinterpret_cast<const float*>(A)) *
                                 (*reinterpret_cast<const float*>(B)) +
                     beta.data.s*(*reinterpret_cast<      float*>(C));
                break;
            case TYPE_DOUBLE:
                *reinterpret_cast<double*>(C) =
                    alpha.data.d*(*reinterpret_cast<const double*>(A)) *
                                 (*reinterpret_cast<const double*>(B)) +
                     beta.data.d*(*reinterpret_cast<      double*>(C));
                break;
            case TYPE_SCOMPLEX:
                *reinterpret_cast<scomplex*>(C) =
                    alpha.data.c*conj(conj_A, *reinterpret_cast<const scomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<const scomplex*>(B)) +
                     beta.data.c*conj(conj_C, *reinterpret_cast<      scomplex*>(C));
                break;
            case TYPE_DCOMPLEX:
                *reinterpret_cast<dcomplex*>(C) =
                    alpha.data.z*conj(conj_A, *reinterpret_cast<const dcomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<const dcomplex*>(B)) +
                     beta.data.z*conj(conj_C, *reinterpret_cast<      dcomplex*>(C));
                break;
        }
    }
}

}
}
