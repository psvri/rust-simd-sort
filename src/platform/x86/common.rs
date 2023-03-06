use core::arch::x86_64::*;
use std::marker::PhantomData;

const X86_SIMD_SORT_INFINITY: f64 = f64::INFINITY;
const X86_SIMD_SORT_INFINITYF: f32 = f32::INFINITY;
const X86_SIMD_SORT_INFINITYH: i16 = 0x7c00;
//const X86_SIMD_SORT_NEGINFINITYH = 0xfc00;
const X86_SIMD_SORT_MAX_UINT16: u16 = u16::MAX;
const X86_SIMD_SORT_MAX_INT16: i16 = i16::MAX;
const X86_SIMD_SORT_MIN_INT16: i16 = i16::MIN;
const X86_SIMD_SORT_MAX_UINT32: u32 = u32::MAX;
const X86_SIMD_SORT_MAX_INT32: i32 = i32::MAX;
const X86_SIMD_SORT_MIN_INT32: i32 = i32::MIN;
const X86_SIMD_SORT_MAX_UINT64: u64 = u64::MAX;
const X86_SIMD_SORT_MAX_INT64: i64 = i64::MAX;
const X86_SIMD_SORT_MIN_INT64: i64 = i64::MIN;
/*const ZMM_MAX_DOUBLE = _mm512_set1_pd(X86_SIMD_SORT_INFINITY);
const ZMM_MAX_UINT64 = _mm512_set1_epi64(X86_SIMD_SORT_MAX_UINT64);
const ZMM_MAX_INT64 = _mm512_set1_epi64(X86_SIMD_SORT_MAX_INT64);
const ZMM_MAX_FLOAT = _mm512_set1_ps(X86_SIMD_SORT_INFINITYF);
const ZMM_MAX_UINT = _mm512_set1_epi32(X86_SIMD_SORT_MAX_UINT32);
const ZMM_MAX_INT = _mm512_set1_epi32(X86_SIMD_SORT_MAX_INT32);
const YMM_MAX_HALF = _mm256_set1_epi16(X86_SIMD_SORT_INFINITYH);
const ZMM_MAX_UINT16 = _mm512_set1_epi16(X86_SIMD_SORT_MAX_UINT16);
const ZMM_MAX_INT16 = _mm512_set1_epi16(X86_SIMD_SORT_MAX_INT16);
SHUFFLE_MASK(a, b, c, d) (a << 6) | (b << 4) | (c << 2) | d*/

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn zmm_max_double() -> __m512d {
    _mm512_set1_pd(X86_SIMD_SORT_INFINITY)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn zmm_max_uint64() -> __m512i {
    _mm512_set1_epi64(X86_SIMD_SORT_MAX_UINT64 as i64)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn zmm_max_int64() -> __m512i {
    _mm512_set1_epi64(X86_SIMD_SORT_MAX_INT64)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn zmm_max_float() -> __m512 {
    _mm512_set1_ps(X86_SIMD_SORT_INFINITYF)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn zmm_max_uint() -> __m512i {
    _mm512_set1_epi32(X86_SIMD_SORT_MAX_UINT32 as i32)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn zmm_max_int() -> __m512i {
    _mm512_set1_epi32(X86_SIMD_SORT_MAX_INT32)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn zmm_max_half() -> __m256i {
    _mm256_set1_epi16(X86_SIMD_SORT_INFINITYH)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn zmm_max_uint16() -> __m512i {
    _mm512_set1_epi16(X86_SIMD_SORT_MAX_UINT16 as i16)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn zmm_max_int16() -> __m512i {
    _mm512_set1_epi16(X86_SIMD_SORT_MAX_INT16)
}

pub trait SimdSortType {
    type NativeType;
    type ZmmT;
    type YmmT;
    type OpmaskT;
    const NUMLANES: u8;

    fn type_max() -> Self::NativeType;
    fn type_min() -> Self::NativeType;
    fn zmm_max() -> Self::ZmmT;
}