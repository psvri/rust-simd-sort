pub mod bit64;

use std::arch::x86_64::__m512i;

use crate::bit_64::qsort_64bit_;

pub fn avx512_sort_i64(data: &mut [i64]) {
    qsort_64bit_::<i64, __m512i>(data, f64::log2(data.len() as f64) as i64)
}

#[cfg(test)]
#[cfg(target_feature="avx512f")]
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
}
