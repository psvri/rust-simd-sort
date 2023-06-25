use std::simd::i64x8;

use crate::bit_64::qsort_64bit_;

pub mod bit_64;

pub fn portable_simd_sort_i64(data: &mut [i64]) {
    qsort_64bit_::<i64, i64x8>(data, f64::log2(data.len() as f64) as i64)
}

#[cfg(test)]
mod test {
    use crate::bit_64::{test::*, *};

    use super::*;

    test_sort_n!(i64, i64x8, 8);
    test_sort_n!(i64, i64x8, 16);
    test_sort_n!(i64, i64x8, 32);
    test_sort_n!(i64, i64x8, 64);
    test_sort_n!(i64, i64x8, 128);
    test_sort_n!(i64, i64x8, 256);
    test_sort_e2e!(i64, i64x8, portable_simd_sort_i64);
}
