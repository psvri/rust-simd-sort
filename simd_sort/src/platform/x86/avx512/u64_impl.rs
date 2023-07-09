use std::{
    arch::x86_64::{
        __m512i, __mmask8, _mm512_cmp_epu64_mask, _mm512_i64gather_epi64, _mm512_loadu_si512,
        _mm512_mask_compressstoreu_epi64, _mm512_mask_loadu_epi64, _mm512_mask_mov_epi64,
        _mm512_mask_storeu_epi64, _mm512_max_epu64, _mm512_min_epu64, _mm512_reduce_max_epu64,
        _mm512_reduce_min_epu64, _mm512_set1_epi64, _mm512_storeu_si512, _MM_CMPINT_NLT,
    },
    mem::transmute,
    slice::from_raw_parts,
};

use crate::{bit_64::Bit64Simd, SimdCompare};

use super::bit_64::{
    network64bit1_idx, network64bit2_idx, network64bit3_idx, network64bit4_idx, permutexvar_m512,
    shuffle_m512, SHUFFLE1_1_1_1, SHUFFLE2_0XAA_MASK, SHUFFLE2_0XCC_MASK, SHUFFLE2_0XF0_MASK,
};

impl SimdCompare<u64, 8> for __m512i {
    type OPMask = __mmask8;

    fn min(a: Self, b: Self) -> Self {
        unsafe { _mm512_min_epu64(a, b) }
    }

    fn max(a: Self, b: Self) -> Self {
        unsafe { _mm512_max_epu64(a, b) }
    }

    fn loadu(data: &[u64]) -> Self {
        unsafe { _mm512_loadu_si512(transmute(data.as_ptr())) }
    }

    fn storeu(input: Self, data: &mut [u64]) {
        unsafe { _mm512_storeu_si512(transmute(data.as_ptr()), input) }
    }

    fn mask_loadu(data: &[u64]) -> Self {
        unsafe {
            let k = (1i32.overflowing_shl(data.len() as u32).0) - 1;
            let max_zmm = Self::set(u64::MAX);
            _mm512_mask_loadu_epi64(max_zmm, k as u8, transmute(data.as_ptr()))
        }
    }

    fn mask_storeu(input: Self, data: &mut [u64]) {
        unsafe {
            let k = (1i32.overflowing_shl(data.len() as u32).0) - 1;
            _mm512_mask_storeu_epi64(transmute(data.as_ptr()), k as u8, input);
        }
    }

    fn gather_from_idx(idx: [usize; 8], data: &[u64]) -> Self {
        unsafe { _mm512_i64gather_epi64(transmute(idx), transmute(data.as_ptr()), 8) }
    }

    fn get_value_at_idx(input: Self, idx: usize) -> u64 {
        unsafe { *from_raw_parts(transmute(&input), 8).get_unchecked(idx) }
    }

    fn set(value: u64) -> Self {
        unsafe { _mm512_set1_epi64(transmute(value)) }
    }

    fn ge(a: Self, b: Self) -> Self::OPMask {
        unsafe { _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_NLT) }
    }

    fn ones_count(mask: Self::OPMask) -> usize {
        mask.count_ones() as usize
    }

    fn not_mask(mask: Self::OPMask) -> Self::OPMask {
        !mask
    }

    fn reducemin(x: Self) -> u64 {
        unsafe { _mm512_reduce_min_epu64(x) }
    }

    fn reducemax(x: Self) -> u64 {
        unsafe { _mm512_reduce_max_epu64(x) }
    }

    fn mask_compressstoreu(array: &mut [u64], mask: Self::OPMask, data: Self) {
        unsafe { _mm512_mask_compressstoreu_epi64(transmute(array.as_ptr()), mask, data) }
    }
}

impl Bit64Simd<u64> for __m512i {
    fn swizzle2_0xaa(a: Self, b: Self) -> Self {
        unsafe { _mm512_mask_mov_epi64(a, SHUFFLE2_0XAA_MASK, b) }
    }

    fn swizzle2_0xcc(a: Self, b: Self) -> Self {
        unsafe { _mm512_mask_mov_epi64(a, SHUFFLE2_0XCC_MASK, b) }
    }

    fn swizzle2_0xf0(a: Self, b: Self) -> Self {
        unsafe { _mm512_mask_mov_epi64(a, SHUFFLE2_0XF0_MASK, b) }
    }

    fn shuffle1_1_1_1(a: Self) -> Self {
        shuffle_m512::<SHUFFLE1_1_1_1>(a)
    }

    fn network64bit1(a: Self) -> Self {
        permutexvar_m512(network64bit1_idx(), a)
    }

    fn network64bit2(a: Self) -> Self {
        permutexvar_m512(network64bit2_idx(), a)
    }

    fn network64bit3(a: Self) -> Self {
        permutexvar_m512(network64bit3_idx(), a)
    }

    fn network64bit4(a: Self) -> Self {
        permutexvar_m512(network64bit4_idx(), a)
    }
}

#[cfg(test)]
#[cfg(target_feature = "avx512f")]
mod test {
    use crate::bit_64::test::*;
    use crate::bit_64::*;
    use crate::platform::x86::avx512::bit_64::test::*;
    use std::slice;

    use super::*;

    fn into_array_u64(x: __m512i) -> [u64; 8] {
        unsafe { slice::from_raw_parts(transmute(&x), 8).try_into().unwrap() }
    }

    test_min_max!(u64, __m512i, into_array_u64);
    test_loadu_storeu!(u64, __m512i, into_array_u64);
    test_mask_loadu_mask_storeu!(u64, __m512i, into_array_u64);
    test_get_at_index!(u64, __m512i);
    test_ge!(u64, __m512i, 0b10101010);
    test_gather!(u64, __m512i, into_array_u64);
    test_not!(u64, __m512i, 0b10101010, !0b10101010);
    test_count_ones!(u64, __m512i, mask_fn);
    test_reduce_min_max!(u64, __m512i);
    test_compress_store_u!(u64, __m512i, u8, generate_mask_answer);
    test_shuffle1_1_1_1!(u64, __m512i, into_array_u64);
    test_swizzle2_0xaa!(u64, __m512i, into_array_u64);
    test_swizzle2_0xcc!(u64, __m512i, into_array_u64);
    test_swizzle2_0xf0!(u64, __m512i, into_array_u64);
    network64bit1!(u64, __m512i, into_array_u64);
    network64bit2!(u64, __m512i, into_array_u64);
    network64bit3!(u64, __m512i, into_array_u64);
    network64bit4!(u64, __m512i, into_array_u64);
}
