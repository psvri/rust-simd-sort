pub mod bit64;

use std::arch::x86_64::__m512i;

use crate::bit_64::qsort_64bit_;

pub fn avx512_sort_i64(data: &mut [i64]) {
    qsort_64bit_::<i64, __m512i>(data, f64::log2(data.len() as f64) as i64)
}

#[cfg(test)]
mod test {
    use crate::bit_64::*;

    use super::*;

    #[test]
    fn test_sort_8() {
        let result: Vec<i64> = (0i64..8).into_iter().collect();
        for i in 0..8 {
            let mut array = Vec::with_capacity(i);
            array.extend_from_slice(&result[..i]);
            array.reverse();
            sort_8::<i64, __m512i>(&mut array);
            assert_eq!(&array, &result[..i]);
        }

        /*let result: Vec<i64> = (0i64..8).into_iter().collect();
        for i in 0..8 {
            let mut array = Vec::with_capacity(i);
            array.extend_from_slice(&result[..i]);
            array.reverse();
            sort_8::<i64, i64x8>(&mut array);
            assert_eq!(&array, &result[..i]);
        }*/
    }

    #[test]
    fn test_sort_16() {
        let result: Vec<i64> = (0i64..16).into_iter().collect();
        for i in 0..16 {
            let mut array = Vec::with_capacity(i);
            array.extend_from_slice(&result[..i]);
            array.reverse();
            sort_16::<i64, __m512i>(&mut array);
            assert_eq!(&array, &result[..i]);
        }

        /* result: Vec<i64> = (0i64..16).into_iter().collect();
        for i in 0..16 {
            let mut array = Vec::with_capacity(i);
            array.extend_from_slice(&result[..i]);
            array.reverse();
            sort_16::<i64, i64x8>(&mut array);
            assert_eq!(&array, &result[..i]);
        }*/
    }

    #[test]
    fn test_sort_32() {
        let result: Vec<i64> = (0i64..32).into_iter().collect();
        for i in 0..32 {
            let mut array = Vec::with_capacity(i);
            array.extend_from_slice(&result[..i]);
            array.reverse();
            sort_32::<i64, __m512i>(&mut array);
            assert_eq!(&array, &result[..i]);
        }

        /*let result: Vec<i64> = (0i64..32).into_iter().collect();
        for i in 0..32 {
            let mut array = Vec::with_capacity(i);
            array.extend_from_slice(&result[..i]);
            array.reverse();
            sort_32::<i64, i64x8>(&mut array);
            assert_eq!(&array, &result[..i]);
        }*/
    }

    #[test]
    fn test_sort_64() {
        let result: Vec<i64> = (0i64..64).into_iter().collect();
        for i in 0..64 {
            let mut array = Vec::with_capacity(i);
            array.extend_from_slice(&result[..i]);
            array.reverse();
            sort_64::<i64, __m512i>(&mut array);
            assert_eq!(&array, &result[..i]);
            println!("succeeded {}", i);
        }
    }

    #[test]
    fn test_sort_128() {
        let result: Vec<i64> = (0i64..128).into_iter().collect();
        for i in 0..128 {
            let mut array = Vec::with_capacity(i);
            array.extend_from_slice(&result[..i]);
            array.reverse();
            sort_128::<i64, __m512i>(&mut array);
            assert_eq!(&array, &result[..i]);
            println!("succeeded {}", i);
        }
    }

    #[test]
    fn test_sort_e2e() {
        let start = 0;
        let end = 1024;
        let result: Vec<i64> = (0..end).into_iter().collect();
        for i in start as usize..end as usize {
            let mut array = Vec::with_capacity(i);
            array.extend_from_slice(&result[..i]);
            array.reverse();
            avx512_sort_i64(array.as_mut_slice());
            assert_eq!(&array, &result[..i]);
            println!("succeeded {}", i);
        }
    }
}
