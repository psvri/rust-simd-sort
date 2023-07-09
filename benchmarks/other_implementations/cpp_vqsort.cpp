#include "hwy/contrib/sort/vqsort.h"
#include "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/contrib/sort/vqsort.cc"

#include <stdint.h>

extern "C"
{
    void vqsort_i64(int64_t *data, size_t len)
    {
        hwy::HWY_NAMESPACE::VQSortStatic(data, len, hwy::SortAscending());
    }

    void vqsort_u64(uint64_t *data, size_t len)
    {
        hwy::HWY_NAMESPACE::VQSortStatic(data, len, hwy::SortAscending());
    }

    void vqsort_f64(double *data, size_t len)
    {
        hwy::HWY_NAMESPACE::VQSortStatic(data, len, hwy::SortAscending());
    }
}