use std::simd::i64x8;

use crate::bit_64::qsort_64bit_;

pub mod bit_64;
pub mod common;

pub fn portable_simd_sort_i64(data: &mut [i64]) {
    qsort_64bit_::<i64, i64x8>(data, f64::log2(data.len() as f64) as i64)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sort_e2e() {
        let start = 0;
        let end = 1024;
        let result: Vec<i64> = (0i64..end).into_iter().collect();
        for i in start..end as usize {
            let mut array = Vec::with_capacity(i);
            array.extend_from_slice(&result[..i]);
            array.reverse();
            portable_simd_sort_i64(&mut array);
            assert_eq!(&array, &result[..i]);
            println!("succeeded {}", i);
        }
    }
}
