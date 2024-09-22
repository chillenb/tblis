#ifndef _TBLIS_PLUGIN_KERNEL_HPP_
#define _TBLIS_PLUGIN_KERNEL_HPP_

#include "tblis.h"
#include "bli_plugin_tblis.h"

namespace tblis
{

// -- Macros and functions to help concisely instantiate reference kernels ---------------------

#define TBLIS_REF_KERNEL(op) PASTECH(op,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX)

#define TBLIS_INSTANTIATE_REF_KERNEL(op,...) \
extern PASTECH(op,_ft) PASTECH(TBLIS_REF_KERNEL(op),_,__LINE__); \
PASTECH(op,_ft) PASTECH(TBLIS_REF_KERNEL(op),_,__LINE__) = &TBLIS_REF_KERNEL(op)<__VA_ARGS__>;

template <typename T>
inline void_fp ptr(T* f)
{
    return reinterpret_cast<void_fp>(f);
}

#define TBLIS_INIT_REF_KERNEL2_(func) func2_t PASTECH(func,_fpa) = [] \
{ \
    func2_t f; \
    bli_func2_init(&f, ptr(func<float,   float>), ptr(func<float,   double>), nullptr                     , nullptr                     , \
                       ptr(func<double,  float>), ptr(func<double,  double>), nullptr                     , nullptr                     , \
                       nullptr                  , nullptr                   , ptr(func<scomplex,scomplex>), ptr(func<scomplex,dcomplex>), \
                       nullptr                  , nullptr                   , ptr(func<dcomplex,scomplex>), ptr(func<dcomplex,dcomplex>)); \
    return f; \
}();
#define TBLIS_INIT_REF_KERNEL2(ker) TBLIS_INIT_REF_KERNEL2_(TBLIS_REF_KERNEL(ker))

#define TBLIS_INIT_REF_KERNEL_(func) func_t PASTECH(func,_fpa) = [] \
{ \
    func_t f; \
    bli_func_init(&f, ptr(func<float>), ptr(func<double>), ptr(func<scomplex>), ptr(func<dcomplex>)); \
    return f; \
}();
#define TBLIS_INIT_REF_KERNEL(ker) TBLIS_INIT_REF_KERNEL_(TBLIS_REF_KERNEL(ker))

#define TBLIS_REF_KERNEL_FPA(ker) PASTECH(TBLIS_REF_KERNEL(ker),_fpa)

//
// Reference kernel function pointers
//

extern func2_t TBLIS_REF_KERNEL_FPA(packm_bsmtc);
extern func_t  TBLIS_REF_KERNEL_FPA(gemm_bsmtc);
extern func_t  TBLIS_REF_KERNEL_FPA(mult);
extern func_t  TBLIS_REF_KERNEL_FPA(reduce);
extern func_t  TBLIS_REF_KERNEL_FPA(shift);
extern func_t  TBLIS_REF_KERNEL_FPA(trans);

//
// Kernel and blocksize IDs
//

#ifndef BLIS_KE_s
#define BLIS_KE_s 8
#endif

#ifndef BLIS_KE_d
#define BLIS_KE_d 4
#endif

#ifndef BLIS_KE_c
#define BLIS_KE_c 4
#endif

#ifndef BLIS_KE_z
#define BLIS_KE_z 2
#endif

#ifndef BLIS_MRT_s
#define BLIS_MRT_s 8
#endif

#ifndef BLIS_MRT_d
#define BLIS_MRT_d 4
#endif

#ifndef BLIS_MRT_c
#define BLIS_MRT_c 4
#endif

#ifndef BLIS_MRT_z
#define BLIS_MRT_z 2
#endif

#ifndef BLIS_NRT_s
#define BLIS_NRT_s 8
#endif

#ifndef BLIS_NRT_d
#define BLIS_NRT_d 4
#endif

#ifndef BLIS_NRT_c
#define BLIS_NRT_c 4
#endif

#ifndef BLIS_NRT_z
#define BLIS_NRT_z 2
#endif

template <typename T> struct KE;
template <> struct KE<float   > : std::integral_constant<len_type,BLIS_KE_s> {};
template <> struct KE<double  > : std::integral_constant<len_type,BLIS_KE_d> {};
template <> struct KE<scomplex> : std::integral_constant<len_type,BLIS_KE_c> {};
template <> struct KE<dcomplex> : std::integral_constant<len_type,BLIS_KE_z> {};

template <typename T> struct MRT;
template <> struct MRT<float   > : std::integral_constant<len_type,BLIS_MRT_s> {};
template <> struct MRT<double  > : std::integral_constant<len_type,BLIS_MRT_d> {};
template <> struct MRT<scomplex> : std::integral_constant<len_type,BLIS_MRT_c> {};
template <> struct MRT<dcomplex> : std::integral_constant<len_type,BLIS_MRT_z> {};

template <typename T> struct NRT;
template <> struct NRT<float   > : std::integral_constant<len_type,BLIS_NRT_s> {};
template <> struct NRT<double  > : std::integral_constant<len_type,BLIS_NRT_d> {};
template <> struct NRT<scomplex> : std::integral_constant<len_type,BLIS_NRT_c> {};
template <> struct NRT<dcomplex> : std::integral_constant<len_type,BLIS_NRT_z> {};

template <typename T> struct MR;
template <> struct MR<float   > : std::integral_constant<len_type,BLIS_MR_s> {};
template <> struct MR<double  > : std::integral_constant<len_type,BLIS_MR_d> {};
template <> struct MR<scomplex> : std::integral_constant<len_type,BLIS_MR_c> {};
template <> struct MR<dcomplex> : std::integral_constant<len_type,BLIS_MR_z> {};

template <typename T> struct NR;
template <> struct NR<float   > : std::integral_constant<len_type,BLIS_NR_s> {};
template <> struct NR<double  > : std::integral_constant<len_type,BLIS_NR_d> {};
template <> struct NR<scomplex> : std::integral_constant<len_type,BLIS_NR_c> {};
template <> struct NR<dcomplex> : std::integral_constant<len_type,BLIS_NR_z> {};

template <typename T> struct BBM;
template <> struct BBM<float   > : std::integral_constant<len_type,BLIS_BBM_s> {};
template <> struct BBM<double  > : std::integral_constant<len_type,BLIS_BBM_d> {};
template <> struct BBM<scomplex> : std::integral_constant<len_type,BLIS_BBM_c> {};
template <> struct BBM<dcomplex> : std::integral_constant<len_type,BLIS_BBM_z> {};

template <typename T> struct BBN;
template <> struct BBN<float   > : std::integral_constant<len_type,BLIS_BBN_s> {};
template <> struct BBN<double  > : std::integral_constant<len_type,BLIS_BBN_d> {};
template <> struct BBN<scomplex> : std::integral_constant<len_type,BLIS_BBN_c> {};
template <> struct BBN<dcomplex> : std::integral_constant<len_type,BLIS_BBN_z> {};

}

#endif
