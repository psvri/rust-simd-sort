use crate::bit_64::qsort_64bit_;

use self::bit_64::Avx2I64x2;

pub mod bit_64;

pub fn avx2_sort_i64(data: &mut [i64]) {
    qsort_64bit_::<i64, Avx2I64x2>(data, f64::log2(data.len() as f64) as i64)
}

#[cfg(test)]
#[cfg(target_feature="avx2")]
mod test {
    use crate::bit_64::{test::*, *};

    use super::*;

    test_sort_n!(i64, Avx2I64x2, 8);
    test_sort_n!(i64, Avx2I64x2, 16);
    test_sort_n!(i64, Avx2I64x2, 32);
    test_sort_n!(i64, Avx2I64x2, 64);
    test_sort_n!(i64, Avx2I64x2, 128);
    test_sort_n!(i64, Avx2I64x2, 256);
    test_sort_e2e!(i64, Avx2I64x2, avx2_sort_i64);
}
