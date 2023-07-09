pub(crate) mod bit_64;
pub(crate) mod f64_impl;
pub(crate) mod i64_impl;
pub(crate) mod u64_impl;

use std::arch::x86_64::{__m512d, __m512i};

use crate::bit_64::qsort_64bit_;

pub fn avx512_sort_i64(data: &mut [i64]) {
    qsort_64bit_::<i64, __m512i>(data, f64::log2(data.len() as f64) as i64)
}

pub fn avx512_sort_u64(data: &mut [u64]) {
    qsort_64bit_::<u64, __m512i>(data, f64::log2(data.len() as f64) as i64)
}

pub fn avx512_sort_f64(data: &mut [f64]) {
    qsort_64bit_::<f64, __m512d>(data, f64::log2(data.len() as f64) as i64)
}

#[cfg(test)]
#[cfg(target_feature = "avx512f")]
mod test {
    use crate::bit_64::{test::*, *};

    use super::*;

    test_sort_n!(i64, __m512i, 8);
    test_sort_n!(i64, __m512i, 16);
    test_sort_n!(i64, __m512i, 32);
    test_sort_n!(i64, __m512i, 64);
    test_sort_n!(i64, __m512i, 128);
    test_sort_n!(i64, __m512i, 256);
    test_sort_e2e!(i64, __m512i, avx512_sort_i64);

    test_sort_n!(u64, __m512i, 8);
    test_sort_n!(u64, __m512i, 16);
    test_sort_n!(u64, __m512i, 32);
    test_sort_n!(u64, __m512i, 64);
    test_sort_n!(u64, __m512i, 128);
    test_sort_n!(u64, __m512i, 256);
    test_sort_e2e!(u64, __m512i, avx512_sort_u64);

    test_sort_n!(f64, __m512d, 8);
    test_sort_n!(f64, __m512d, 16);
    test_sort_n!(f64, __m512d, 32);
    test_sort_n!(f64, __m512d, 64);
    test_sort_n!(f64, __m512d, 128);
    test_sort_n!(f64, __m512d, 256);
    test_sort_e2e!(f64, __m512i, avx512_sort_f64);
}
