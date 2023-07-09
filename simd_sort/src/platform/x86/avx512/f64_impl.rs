use std::{
    arch::x86_64::{
        __m512d, __m512i, __mmask8, _mm512_cmp_pd_mask, _mm512_i64gather_pd, _mm512_loadu_pd,
        _mm512_mask_compressstoreu_pd, _mm512_mask_loadu_pd, _mm512_mask_mov_pd,
        _mm512_mask_storeu_pd, _mm512_max_pd, _mm512_min_pd, _mm512_permutexvar_pd,
        _mm512_reduce_max_pd, _mm512_reduce_min_pd, _mm512_set1_pd, _mm512_shuffle_pd,
        _mm512_storeu_pd, _CMP_GE_OQ, _MM_PERM_ENUM,
    },
    mem::transmute,
    slice::from_raw_parts,
};

use crate::{bit_64::Bit64Simd, SimdCompare};

use super::bit_64::{
    network64bit1_idx, network64bit2_idx, network64bit3_idx, network64bit4_idx, SHUFFLE1_1_1_1,
    SHUFFLE2_0XAA_MASK, SHUFFLE2_0XCC_MASK, SHUFFLE2_0XF0_MASK,
};

fn permutexvar_m512d(idx: __m512i, a: __m512d) -> __m512d {
    unsafe { _mm512_permutexvar_pd(idx, a) }
}

fn shuffle_m512d<const MASK: _MM_PERM_ENUM>(zmm: __m512d) -> __m512d {
    unsafe { _mm512_shuffle_pd::<MASK>(zmm, zmm) }
}

impl SimdCompare<f64, 8> for __m512d {
    type OPMask = __mmask8;

    fn min(a: Self, b: Self) -> Self {
        unsafe { _mm512_min_pd(a, b) }
    }

    fn max(a: Self, b: Self) -> Self {
        unsafe { _mm512_max_pd(a, b) }
    }

    fn loadu(data: &[f64]) -> Self {
        unsafe { _mm512_loadu_pd(transmute(data.as_ptr())) }
    }

    fn storeu(input: Self, data: &mut [f64]) {
        unsafe { _mm512_storeu_pd(transmute(data.as_ptr()), input) }
    }

    fn mask_loadu(data: &[f64]) -> Self {
        unsafe {
            let k = (1i32.overflowing_shl(data.len() as u32).0) - 1;
            let max_zmm = Self::set(f64::MAX);
            _mm512_mask_loadu_pd(max_zmm, k as u8, transmute(data.as_ptr()))
        }
    }

    fn mask_storeu(input: Self, data: &mut [f64]) {
        unsafe {
            let k = (1i32.overflowing_shl(data.len() as u32).0) - 1;
            _mm512_mask_storeu_pd(transmute(data.as_ptr()), k as u8, input);
        }
    }

    fn gather_from_idx(idx: [usize; 8], data: &[f64]) -> Self {
        unsafe { _mm512_i64gather_pd(transmute(idx), transmute(data.as_ptr()), 8) }
    }

    fn get_value_at_idx(input: Self, idx: usize) -> f64 {
        unsafe { *from_raw_parts(transmute(&input), 8).get_unchecked(idx) }
    }

    fn set(value: f64) -> Self {
        unsafe { _mm512_set1_pd(value) }
    }

    fn ge(a: Self, b: Self) -> Self::OPMask {
        unsafe { _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ) }
    }

    fn ones_count(mask: Self::OPMask) -> usize {
        mask.count_ones() as usize
    }

    fn not_mask(mask: Self::OPMask) -> Self::OPMask {
        !mask
    }

    fn reducemin(x: Self) -> f64 {
        unsafe { _mm512_reduce_min_pd(x) }
    }

    fn reducemax(x: Self) -> f64 {
        unsafe { _mm512_reduce_max_pd(x) }
    }

    fn mask_compressstoreu(array: &mut [f64], mask: Self::OPMask, data: Self) {
        unsafe { _mm512_mask_compressstoreu_pd(transmute(array.as_ptr()), mask, data) }
    }
}

impl Bit64Simd<f64> for __m512d {
    fn swizzle2_0xaa(a: Self, b: Self) -> Self {
        unsafe { _mm512_mask_mov_pd(a, SHUFFLE2_0XAA_MASK, b) }
    }

    fn swizzle2_0xcc(a: Self, b: Self) -> Self {
        unsafe { _mm512_mask_mov_pd(a, SHUFFLE2_0XCC_MASK, b) }
    }

    fn swizzle2_0xf0(a: Self, b: Self) -> Self {
        unsafe { _mm512_mask_mov_pd(a, SHUFFLE2_0XF0_MASK, b) }
    }

    fn shuffle1_1_1_1(a: Self) -> Self {
        shuffle_m512d::<SHUFFLE1_1_1_1>(a)
    }

    fn network64bit1(a: Self) -> Self {
        permutexvar_m512d(network64bit1_idx(), a)
    }

    fn network64bit2(a: Self) -> Self {
        permutexvar_m512d(network64bit2_idx(), a)
    }

    fn network64bit3(a: Self) -> Self {
        permutexvar_m512d(network64bit3_idx(), a)
    }

    fn network64bit4(a: Self) -> Self {
        permutexvar_m512d(network64bit4_idx(), a)
    }
}

#[cfg(test)]
#[cfg(target_feature = "avx512f")]
pub mod test {
    use crate::{bit_64::test::*, platform::x86::avx512::bit_64::test::*};
    use std::slice::from_raw_parts;

    use super::*;

    fn into_array_f64(x: __m512d) -> [f64; 8] {
        unsafe { from_raw_parts(transmute(&x), 8).try_into().unwrap() }
    }

    test_min_max!(f64, __m512d, into_array_f64);
    test_loadu_storeu!(f64, __m512d, into_array_f64);
    test_mask_loadu_mask_storeu!(f64, __m512d, into_array_f64);
    test_get_at_index!(f64, __m512d);
    test_ge!(f64, __m512d, 0b10101010);
    test_gather!(f64, __m512d, into_array_f64);
    test_not!(f64, __m512d, 0b10101010, !0b10101010);
    test_count_ones!(f64, __m512d, mask_fn);
    test_reduce_min_max!(f64, __m512d);
    test_compress_store_u!(f64, __m512d, u8, generate_mask_answer);
    test_shuffle1_1_1_1!(f64, __m512d, into_array_f64);
    test_swizzle2_0xaa!(f64, __m512d, into_array_f64);
    test_swizzle2_0xcc!(f64, __m512d, into_array_f64);
    test_swizzle2_0xf0!(f64, __m512d, into_array_f64);
    network64bit1!(f64, __m512d, into_array_f64);
    network64bit2!(f64, __m512d, into_array_f64);
    network64bit3!(f64, __m512d, into_array_f64);
    network64bit4!(f64, __m512d, into_array_f64);
}
