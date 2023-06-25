#include "hwy/contrib/sort/vqsort_i64a.cc"
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/contrib/sort/vqsort-inl.h"
#include "hwy/contrib/sort/vqsort.cc"

#include <stdint.h>

extern "C"
{
    void vqsort_i64(int64_t *data, size_t len)
    {
        hwy::HWY_NAMESPACE::SortI64Asc(data, len);
    }
}