use crate::bit_64::qsort_64bit_;

use self::bit_64::Avx2I64x2;

pub mod bit_64;

pub fn avx2_sort_i64(data: &mut [i64]) {
    qsort_64bit_::<i64, Avx2I64x2>(data, f64::log2(data.len() as f64) as i64)
}

#[cfg(test)]
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

    // #[test]
    // fn test_sort_8() {
    //     let result: Vec<i64> = (0i64..8).into_iter().collect();
    //     for i in 0..8 {
    //         let mut array = Vec::with_capacity(i);
    //         array.extend_from_slice(&result[..i]);
    //         array.reverse();
    //         sort_8::<i64, Avx2I64x2>(&mut array);
    //         assert_eq!(&array, &result[..i]);
    //     }

    //     /*let result: Vec<i64> = (0i64..8).into_iter().collect();
    //     for i in 0..8 {
    //         let mut array = Vec::with_capacity(i);
    //         array.extend_from_slice(&result[..i]);
    //         array.reverse();
    //         sort_8::<i64, i64x8>(&mut array);
    //         assert_eq!(&array, &result[..i]);
    //     }*/
    // }

    // #[test]
    // fn test_sort_16() {
    //     let result: Vec<i64> = (0i64..16).into_iter().collect();
    //     for i in 0..16 {
    //         let mut array = Vec::with_capacity(i);
    //         array.extend_from_slice(&result[..i]);
    //         array.reverse();
    //         sort_16::<i64, Avx2I64x2>(&mut array);
    //         assert_eq!(&array, &result[..i]);
    //     }

    //     /* result: Vec<i64> = (0i64..16).into_iter().collect();
    //     for i in 0..16 {
    //         let mut array = Vec::with_capacity(i);
    //         array.extend_from_slice(&result[..i]);
    //         array.reverse();
    //         sort_16::<i64, i64x8>(&mut array);
    //         assert_eq!(&array, &result[..i]);
    //     }*/
    // }

    // #[test]
    // fn test_sort_32() {
    //     let result: Vec<i64> = (0i64..32).into_iter().collect();
    //     for i in 0..32 {
    //         let mut array = Vec::with_capacity(i);
    //         array.extend_from_slice(&result[..i]);
    //         array.reverse();
    //         sort_32::<i64, Avx2I64x2>(&mut array);
    //         assert_eq!(&array, &result[..i]);
    //     }

    //     /*let result: Vec<i64> = (0i64..32).into_iter().collect();
    //     for i in 0..32 {
    //         let mut array = Vec::with_capacity(i);
    //         array.extend_from_slice(&result[..i]);
    //         array.reverse();
    //         sort_32::<i64, i64x8>(&mut array);
    //         assert_eq!(&array, &result[..i]);
    //     }*/
    // }

    // #[test]
    // fn test_sort_64() {
    //     let result: Vec<i64> = (0i64..64).into_iter().collect();
    //     for i in 0..64 {
    //         let mut array = Vec::with_capacity(i);
    //         array.extend_from_slice(&result[..i]);
    //         array.reverse();
    //         sort_64::<i64, Avx2I64x2>(&mut array);
    //         assert_eq!(&array, &result[..i]);
    //         println!("succeeded {}", i);
    //     }
    // }

    // #[test]
    // fn test_sort_128() {
    //     let result: Vec<i64> = (0i64..128).into_iter().collect();
    //     for i in 0..128 {
    //         let mut array = Vec::with_capacity(i);
    //         array.extend_from_slice(&result[..i]);
    //         array.reverse();
    //         sort_128::<i64, Avx2I64x2>(&mut array);
    //         assert_eq!(&array, &result[..i]);
    //         println!("succeeded {}", i);
    //     }
    // }

    // #[test]
    // fn test_sort_256() {
    //     let start = 0;
    //     let result: Vec<i64> = (0i64..256).into_iter().collect();
    //     for i in start..256 {
    //         let mut array = Vec::with_capacity(i);
    //         array.extend_from_slice(&result[..i]);
    //         array.reverse();
    //         sort_256::<i64, Avx2I64x2>(&mut array);
    //         assert_eq!(&array, &result[..i]);
    //         println!("succeeded {}", i);
    //     }
    // }

    // #[test]
    // fn test_sort_e2e() {
    //     let start = 0;
    //     let end = 1024;
    //     let result: Vec<i64> = (0..end).into_iter().collect();
    //     for i in start as usize..end as usize {
    //         let mut array = Vec::with_capacity(i);
    //         array.extend_from_slice(&result[..i]);
    //         array.reverse();
    //         avx2_sort_i64(array.as_mut_slice());
    //         assert_eq!(&array, &result[..i]);
    //         println!("succeeded {}", i);
    //     }
    // }
}
