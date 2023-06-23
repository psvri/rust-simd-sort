use core::slice;
use std::{
    arch::x86_64::{
        __m512i, __mmask8, _mm512_castpd_si512, _mm512_castsi512_pd, _mm512_cmp_epi64_mask,
        _mm512_i64gather_epi64, _mm512_loadu_si512, _mm512_mask_compressstoreu_epi64,
        _mm512_mask_loadu_epi64, _mm512_mask_mov_epi64, _mm512_mask_storeu_epi64, _mm512_max_epi64,
        _mm512_min_epi64, _mm512_permutexvar_epi64, _mm512_reduce_max_epi64,
        _mm512_reduce_min_epi64, _mm512_set1_epi64, _mm512_set_epi64, _mm512_shuffle_pd,
        _mm512_storeu_si512, _MM_CMPINT_NLT, _MM_PERM_ENUM,
    },
    mem,
};

use crate::{bit_64::Bit64Simd, SimdCompare};

impl SimdCompare<i64, 8> for __m512i {
    type OPMask = __mmask8;

    fn min(a: Self, b: Self) -> Self {
        unsafe { _mm512_min_epi64(a, b) }
    }

    fn max(a: Self, b: Self) -> Self {
        unsafe { _mm512_max_epi64(a, b) }
    }

    fn loadu(data: &[i64]) -> Self {
        unsafe { _mm512_loadu_si512(mem::transmute(data.as_ptr())) }
    }

    fn storeu(input: Self, data: &mut [i64]) {
        unsafe { _mm512_storeu_si512(mem::transmute(data.as_ptr()), input) }
    }

    fn mask_loadu(data: &[i64]) -> Self {
        unsafe {
            let k = (1i32.overflowing_shl(data.len() as u32).0) - 1;
            let max_zmm = Self::set(i64::MAX);
            _mm512_mask_loadu_epi64(max_zmm, k as u8, mem::transmute(data.as_ptr()))
        }
    }

    fn mask_storeu(input: Self, data: &mut [i64]) {
        unsafe {
            let k = (1i32.overflowing_shl(data.len() as u32).0) - 1;
            _mm512_mask_storeu_epi64(mem::transmute(data.as_ptr()), k as u8, input);
        }
    }

    fn gather_from_idx(idx: [usize; 8], data: &[i64]) -> Self {
        unsafe { _mm512_i64gather_epi64(mem::transmute(idx), mem::transmute(data.as_ptr()), 8) }
    }

    fn get_value_at_idx(input: Self, idx: usize) -> i64 {
        unsafe { slice::from_raw_parts(mem::transmute(&input), 8)[idx] }
    }

    fn set(value: i64) -> Self {
        unsafe { _mm512_set1_epi64(value) }
    }

    fn ge(a: Self, b: Self) -> Self::OPMask {
        unsafe { _mm512_cmp_epi64_mask::<_MM_CMPINT_NLT>(a, b) }
    }

    fn ones_count(mask: Self::OPMask) -> usize {
        mask.count_ones() as usize
    }

    fn not_mask(mask: Self::OPMask) -> Self::OPMask {
        !mask
    }

    fn reducemin(x: Self) -> i64 {
        unsafe { _mm512_reduce_min_epi64(x) }
    }

    fn reducemax(x: Self) -> i64 {
        unsafe { _mm512_reduce_max_epi64(x) }
    }

    fn mask_compressstoreu(array: &mut [i64], mask: Self::OPMask, data: Self) {
        unsafe { _mm512_mask_compressstoreu_epi64(mem::transmute(array.as_ptr()), mask, data) }
    }
}

const SHUFFLE1_1_1_1: _MM_PERM_ENUM = shuffle_mask([1, 1, 1, 1]);

const fn shuffle_mask(a: [_MM_PERM_ENUM; 4]) -> _MM_PERM_ENUM {
    (a[0] << 6) | (a[1] << 4) | (a[2] << 2) | a[3]
}

fn shuffle_m512<const MASK: _MM_PERM_ENUM>(zmm: __m512i) -> __m512i {
    unsafe {
        let temp = _mm512_castsi512_pd(zmm);
        _mm512_castpd_si512(_mm512_shuffle_pd::<MASK>(temp, temp))
    }
}

fn permutexvar_m512(idx: __m512i, a: __m512i) -> __m512i {
    unsafe { _mm512_permutexvar_epi64(idx, a) }
}

impl Bit64Simd<i64> for __m512i {
    fn swizzle2_0xaa(a: Self, b: Self) -> Self {
        unsafe { _mm512_mask_mov_epi64(a, 0xAA, b) }
    }

    fn swizzle2_0xcc(a: Self, b: Self) -> Self {
        unsafe { _mm512_mask_mov_epi64(a, 0xCC, b) }
    }

    fn swizzle2_0xf0(a: Self, b: Self) -> Self {
        unsafe { _mm512_mask_mov_epi64(a, 0xF0, b) }
    }

    fn shuffle1_1_1_1(a: Self) -> Self {
        shuffle_m512::<SHUFFLE1_1_1_1>(a)
    }

    fn network64bit1(a: Self) -> Self {
        unsafe { permutexvar_m512(_mm512_set_epi64(4, 5, 6, 7, 0, 1, 2, 3), a) }
    }

    fn network64bit2(a: Self) -> Self {
        unsafe { permutexvar_m512(_mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7), a) }
    }

    fn network64bit3(a: Self) -> Self {
        unsafe { permutexvar_m512(_mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2), a) }
    }

    fn network64bit4(a: Self) -> Self {
        unsafe { permutexvar_m512(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), a) }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn from_array_for_m512(data: [i64; 8]) -> __m512i {
        unsafe { _mm512_loadu_si512(mem::transmute(data.as_ptr())) }
    }

    fn into_array_for_m512(x: __m512i) -> [i64; 8] {
        unsafe {
            slice::from_raw_parts(mem::transmute(&x), 8)
                .try_into()
                .unwrap()
        }
    }

    #[test]
    fn test_min_max() {
        let first = from_array_for_m512([1, 20, 3, 40, 5, 60, 70, 80]);
        let second = from_array_for_m512([10, 2, 30, 4, 50, 6, 7, 8]);
        assert_eq!(
            into_array_for_m512(<__m512i as SimdCompare<i64, 8>>::min(first, second)),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(
            into_array_for_m512(<__m512i as SimdCompare<i64, 8>>::max(first, second)),
            [10, 20, 30, 40, 50, 60, 70, 80]
        );
    }

    #[test]
    fn test_loadu_storeu() {
        let mut input_slice = [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let first = __m512i::loadu(input_slice.as_ref());
        assert_eq!(into_array_for_m512(first), [1, 2, 3, 4, 5, 6, 7, 8]);
        __m512i::storeu(first, &mut input_slice[2..]);
        assert_eq!(input_slice, [1i64, 2, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_mask_loadu_mask_storeu() {
        let mut input_slice = [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let first = __m512i::mask_loadu(&input_slice[..2]);
        assert_eq!(
            into_array_for_m512(first),
            [
                1,
                2,
                i64::MAX,
                i64::MAX,
                i64::MAX,
                i64::MAX,
                i64::MAX,
                i64::MAX
            ]
        );
        __m512i::mask_storeu(first, &mut input_slice[2..4]);
        assert_eq!(input_slice, [1i64, 2, 1, 2, 5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_get_at_index() {
        let first = from_array_for_m512([1, 2, 3, 4, 5, 6, 7, 8]);
        for i in 1..9 {
            assert_eq!(i as i64, __m512i::get_value_at_idx(first, i - 1));
        }
    }

    #[test]
    fn test_ge() {
        let first = from_array_for_m512([1, 20, 3, 40, 5, 60, 7, 80]);
        let second = from_array_for_m512([10, 2, 30, 40, 50, 6, 70, 80]);
        let result_mask = __m512i::ge(first, second);
        assert_eq!(result_mask, 0b10101010);
    }

    #[test]
    fn test_gather() {
        let input_slice = [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let first = __m512i::gather_from_idx([1, 1, 2, 2, 9, 9, 5, 6], input_slice.as_ref());
        assert_eq!(into_array_for_m512(first), [2, 2, 3, 3, 10, 10, 6, 7]);
    }

    #[test]
    fn test_not() {
        let first: __mmask8 = 10;
        assert_eq!(__m512i::not_mask(first), !first);
    }

    #[test]
    fn test_reduce_min_max() {
        let first = from_array_for_m512([5, 6, 3, 4, 1, 2, 9, 8]);
        assert_eq!(__m512i::reducemin(first), 1);
        assert_eq!(__m512i::reducemax(first), 9);
    }

    fn generate_mask_answer(bitmask: usize, values: &[i64]) -> [i64; 8] {
        let mut new_values = [0; 8];
        let mut count = 0;
        for i in 0..8 {
            if bitmask & (1 << i) != 0 {
                new_values[count] = values[i];
                count += 1;
            }
        }
        new_values
    }

    #[test]
    fn test_compress_store_u() {
        let input_slice = [1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let first = __m512i::loadu(input_slice.as_ref());
        for i in 0..255 {
            let new_values = generate_mask_answer(dbg!(i), &input_slice);
            let mask = i;
            dbg!(format!("{:?}", mask));
            let mut new_array = input_slice.clone();
            __m512i::mask_compressstoreu(&mut new_array[2..], mask as u8, first);
            dbg!(format!("{:?}", new_array));
            dbg!(format!("{:?}", new_values));
            println!("{:?}", new_array);
            for j in 0..i.count_ones() as usize {
                assert_eq!(new_array[2 + j], new_values[j]);
            }
            for j in i..8 {
                assert_eq!(new_array[2 + j], input_slice[2 + j]);
            }
        }
    }

    #[test]
    fn test_shuffle1_1_1_1() {
        let first = from_array_for_m512([1i64, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            into_array_for_m512(__m512i::shuffle1_1_1_1(first)),
            [2, 1, 4, 3, 6, 5, 8, 7]
        );
    }

    #[test]
    fn test_count_ones() {
        for i in 0u8..8 {
            let mask = i as <std::arch::x86_64::__m512i as SimdCompare<i64, 8>>::OPMask;
            assert_eq!(__m512i::ones_count(mask), i.count_ones() as usize);
        }
    }

    #[test]
    fn test_swizzle2_0xaa() {
        let first = from_array_for_m512([1i64, 2, 3, 4, 5, 6, 7, 8]);
        let second = from_array_for_m512([10i64, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(
            into_array_for_m512(__m512i::swizzle2_0xaa(first, second)),
            [1, 20, 3, 40, 5, 60, 7, 80]
        );
    }

    #[test]
    fn test_swizzle2_0xcc() {
        let first = from_array_for_m512([1, 2, 3, 4, 5, 6, 7, 8]);
        let second = from_array_for_m512([10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(
            into_array_for_m512(__m512i::swizzle2_0xcc(first, second)),
            [1, 2, 30, 40, 5, 6, 70, 80]
        );
    }

    #[test]
    fn test_swizzle2_0xf0() {
        let first = from_array_for_m512([1, 2, 3, 4, 5, 6, 7, 8]);
        let second = from_array_for_m512([10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(
            into_array_for_m512(__m512i::swizzle2_0xf0(first, second)),
            [1, 2, 3, 4, 50, 60, 70, 80]
        );
    }
}
