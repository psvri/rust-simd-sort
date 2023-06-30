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
#[cfg(target_feature="avx512f")]
mod test {
    use crate::bit_64::test::*;
    use std::ops::*;

    use super::*;

    fn into_array_i64(x: __m512i) -> [i64; 8] {
        unsafe {
            slice::from_raw_parts(mem::transmute(&x), 8)
                .try_into()
                .unwrap()
        }
    }

    fn generate_mask_answer<T, M>(bitmask: M, values: &[T]) -> (M, [T; 8])
    where
        T: TryFrom<usize> + Default + Copy,
        M: BitAnd<u8> + std::marker::Copy,
        <M as BitAnd<u8>>::Output: PartialEq<u8>,
    {
        let mut new_values = [<T as Default>::default(); 8];
        let mut count = 0;
        for i in 0..8 {
            if bitmask & (1 << i) != 0 {
                new_values[count] = values[i];
                count += 1;
            }
        }
        (bitmask, new_values)
    }

    fn mask_fn(x: u8) -> u8 {
        x
    }

    test_min_max!(i64, __m512i, into_array_i64);
    test_loadu_storeu!(i64, __m512i, into_array_i64);
    test_mask_loadu_mask_storeu!(i64, __m512i, into_array_i64);
    test_get_at_index!(i64, __m512i);
    test_ge!(i64, __m512i, 0b10101010);
    test_gather!(i64, __m512i, into_array_i64);
    test_not!(i64, __m512i, 0b10101010, !0b10101010);
    test_count_ones!(i64, __m512i, mask_fn);
    test_reduce_min_max!(i64, __m512i);
    test_compress_store_u!(i64, __m512i, u8, generate_mask_answer);
    test_shuffle1_1_1_1!(i64, __m512i, into_array_i64);
    test_swizzle2_0xaa!(i64, __m512i, into_array_i64);
    test_swizzle2_0xcc!(i64, __m512i, into_array_i64);
    test_swizzle2_0xf0!(i64, __m512i, into_array_i64);
    network64bit1!(i64, __m512i, into_array_i64);
    network64bit2!(i64, __m512i, into_array_i64);
    network64bit3!(i64, __m512i, into_array_i64);
    network64bit4!(i64, __m512i, into_array_i64);
}
