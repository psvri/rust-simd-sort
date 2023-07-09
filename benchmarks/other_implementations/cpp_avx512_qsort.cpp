#include <avx512-common-qsort.h>
#include <avx512-64bit-qsort.hpp>

#include <stdint.h>

extern "C"
{
    void avx512_qsort_i64(int64_t *data, size_t len)
    {
        avx512_qsort<int64_t>(data, len);
    }

    void avx512_qsort_u64(uint64_t *data, size_t len)
    {
        avx512_qsort<uint64_t>(data, len);
    }

    void avx512_qsort_f64(double *data, size_t len)
    {
        avx512_qsort<double>(data, len);
    }
}