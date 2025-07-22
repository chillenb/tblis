#ifndef _TBLIS_ALIGNMENT_HPP_
#define _TBLIS_ALIGNMENT_HPP_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cmath>

#include "basic_types.h"

#if defined(TBLIS_HAVE_GCC_BITSET_BUILTINS)

template <typename T>
int countr_zero(T x)
{
    if constexpr (sizeof(T) == sizeof(unsigned long long))
    {
        return __builtin_ctzll(static_cast<unsigned long long>(x));
    }
    else if constexpr (sizeof(T) == sizeof(unsigned long))
    {
        return __builtin_ctzll(static_cast<unsigned long>(x));
    }
    else
    {
        return __builtin_ctzll(static_cast<unsigned>(x));
    }
}

#elif defined(TBLIS_HAVE_CXX20_BITSET)

#include <bit>

template <typename T>
int countr_zero(T x)
{
    return std::countr_zero(static_cast<std::make_unsigned_t<T>>(x));
}

#else

template <typename T>
int countr_zero(T x_)
{
    auto x = static_cast<std::make_unsigned_t<T>>(x_);
    auto i = 0;
    for (auto b = 1;i < sizeof(x)*CHAR_BIT;i++, b <<= 1)
        if (x & b) return i;
    return i;
}

#endif

namespace tblis
{

template <typename T>
T gcd(T a_, T b_)
{
    auto a = static_cast<std::make_unsigned_t<T>>(std::abs(a_));
    auto b = static_cast<std::make_unsigned_t<T>>(std::abs(b_));

    if (a == 0) return b;
    if (b == 0) return a;

    unsigned d = countr_zero(a|b);

    a >>= countr_zero(a);
    b >>= countr_zero(b);

    while (a != b)
    {
        if (a > b)
        {
            a = (a-b)>>1;
        }
        else
        {
            b = (b-a)>>1;
        }
    }

    return a<<d;
}

template <typename T>
T lcm(T a, T b)
{
    return a*(b/gcd(a,b));
}

template <typename T, typename U>
constexpr std::common_type_t<T,U> remainder(T N, U B)
{
    return (B-1)-(N+B-1)%B;
}

template <typename T, typename U>
constexpr std::common_type_t<T,U> round_up(T N, U B)
{
    return N + remainder(N, B);
}

template <typename T, typename U>
constexpr std::common_type_t<T,U> ceil_div(T N, U D)
{
    return (N >= 0 ? (N+D-1)/D : (N-D+1)/D);
}

template <typename T, typename U>
constexpr std::common_type_t<T,U> floor_div(T N, U D)
{
    return N/D;
}

template <typename U, typename T>
U* convert_and_align(T* x)
{
    uintptr_t off = (reinterpret_cast<uintptr_t>(x))%alignof(U);
    return reinterpret_cast<U*>(reinterpret_cast<char*>(x) + (off == 0 ? 0 : alignof(U)-off));
}

template <typename T, typename U>
constexpr size_t size_as_type(size_t n)
{
    return ceil_div(n*sizeof(T) + alignof(T), sizeof(U));
}

template <typename T>
constexpr size_t size_as_type(size_t n, type_t type)
{
    return ceil_div(n*sizeof(T) + alignof(T), type_size[type]);
}

}

#endif
