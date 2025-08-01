#include "../test.hpp"

/*
 * Creates a random matrix times vector operation, where each matrix
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_gemv(stride_type N, matrix<T>& A,
                                matrix<T>& B,
                                matrix<T>& C)
{
    len_type m = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));
    len_type k = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));

    random_matrix(N, m, k, A);
    random_matrix(N, k, 1, B);
    random_matrix(N, m, 1, C);
}

REPLICATED_TEMPLATED_TEST_CASE(gemv, R, T, all_types)
{
    matrix<T> A, B, C, D, E;

    random_gemv(N/100, A, B, C);

    T scale(10.0*random_unit<T>());

    len_type m = C.length(0);
    len_type n = C.length(1);
    len_type k = A.length(1);

    INFO_OR_PRINT("m, n, k    = " << m << ", " << n << ", " << k);
    INFO_OR_PRINT("rs_a, cs_a = " << A.stride(0) << ", " << A.stride(1));
    INFO_OR_PRINT("rs_b, cs_b = " << B.stride(0) << ", " << B.stride(1));
    INFO_OR_PRINT("rs_c, cs_c = " << C.stride(0) << ", " << C.stride(1));

    D.reset(C);
    gemm_ref<T>(scale, A, B, scale, D);

    E.reset(C);
    mult(scale, A, B, scale, E);

    add(-1, D, 1, E);
    T error = reduce<T>(REDUCE_NORM_2, E);

    check("REF", error, scale*m*n*k);
}
