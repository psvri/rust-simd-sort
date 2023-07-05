use std::arch::x86_64::{
    __m256d, _mm256_blendv_pd, _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_cmp_pd,
    _mm256_extractf128_pd, _mm256_i64gather_pd, _mm256_loadu_pd, _mm256_loadu_si256,
    _mm256_mask_i32gather_pd, _mm256_maskstore_pd, _mm256_max_pd, _mm256_min_pd,
    _mm256_movemask_pd, _mm256_permute4x64_pd, _mm256_permutevar8x32_epi32, _mm256_set1_pd,
    _mm256_shuffle_pd, _mm256_storeu_pd, _mm256_xor_pd, _mm_max_pd, _mm_min_pd, _mm_permute_pd,
    _CMP_GE_OQ,
};
use std::{mem, slice};

use crate::bit_64::Bit64Simd;
use crate::SimdCompare;

use super::bit_64::{
    COMPRESS_MASK, COMPRESS_PERMUTATIONS, LOADU_MASK, NETWORK_64BIT_1, NETWORK_64BIT_2,
    NETWORK_64BIT_3, SHUFFLE1_1_1_1,
};

const V_INDEX_1: [i32; 4] = [0, 1, 2, 3];
const V_INDEX_2: [i32; 4] = [4, 5, 6, 7];

#[derive(Debug, Copy, Clone)]
pub struct Avx2F64x2 {
    values: [__m256d; 2],
}

impl PartialEq for Avx2F64x2 {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let a_ptr: *const i64 = &self.values as *const _ as *const i64;
            let b_ptr: *const i64 = &other.values as *const _ as *const i64;
            let a_slice = std::slice::from_raw_parts(a_ptr, 8);
            let b_slice = std::slice::from_raw_parts(b_ptr, 8);

            a_slice == b_slice
        }
    }
}

impl From<[f64; 8]> for Avx2F64x2 {
    fn from(v: [f64; 8]) -> Self {
        unsafe { mem::transmute(v) }
    }
}

impl Avx2F64x2 {
    fn as_slice(&self) -> &[f64] {
        unsafe { slice::from_raw_parts(mem::transmute(self.values.as_ptr()), 8) }
    }
}

#[inline]
fn blend_256d(a: __m256d, b: __m256d, mask: __m256d) -> __m256d {
    unsafe { _mm256_blendv_pd(a, b, mask) }
}

#[inline]
fn shuffle_256d<const MASK: i32>(a: __m256d, b: __m256d) -> __m256d {
    unsafe { _mm256_shuffle_pd(a, b, MASK) }
}

impl SimdCompare<f64, 8> for Avx2F64x2 {
    type OPMask = Self;

    fn min(a: Self, b: Self) -> Self {
        unsafe {
            let value1 = _mm256_min_pd(a.values[0], b.values[0]);
            let value2 = _mm256_min_pd(a.values[1], b.values[1]);
            return Self {
                values: [value1, value2],
            };
        }
    }

    fn max(a: Self, b: Self) -> Self {
        unsafe {
            let value1 = _mm256_max_pd(a.values[0], b.values[0]);
            let value2 = _mm256_max_pd(a.values[1], b.values[1]);
            return Self {
                values: [value1, value2],
            };
        }
    }

    fn loadu(data: &[f64]) -> Self {
        unsafe {
            let base_ptr = data.as_ptr();
            let v1 = _mm256_loadu_pd(base_ptr);
            let v2 = _mm256_loadu_pd(base_ptr.offset(4));
            Self { values: [v1, v2] }
        }
    }

    fn storeu(input: Self, data: &mut [f64]) {
        unsafe {
            let base_ptr = data.as_mut_ptr();
            _mm256_storeu_pd(base_ptr, input.values[0]);
            _mm256_storeu_pd(base_ptr.offset(4), input.values[1]);
        }
    }

    fn mask_loadu(data: &[f64]) -> Self {
        unsafe {
            let mask = LOADU_MASK.get_unchecked(data.len()).as_ptr();
            let mask1 = _mm256_loadu_pd(mem::transmute(mask));
            let mask2 = _mm256_loadu_pd(mem::transmute(mask.offset(4)));
            let indices1 = mem::transmute(V_INDEX_1);
            let indices2 = mem::transmute(V_INDEX_2);
            let max_values = _mm256_set1_pd(f64::MAX);
            let v1 = _mm256_mask_i32gather_pd(max_values, data.as_ptr(), indices1, mask1, 8);
            let v2 = _mm256_mask_i32gather_pd(max_values, data.as_ptr(), indices2, mask2, 8);
            Self { values: [v1, v2] }
        }
    }

    fn mask_storeu(input: Self, data: &mut [f64]) {
        unsafe {
            let mask = LOADU_MASK.get_unchecked(data.len());
            let mask1 = _mm256_loadu_si256(mem::transmute(mask.as_ptr()));
            let mask2 = _mm256_loadu_si256(mem::transmute(mask[4..].as_ptr()));
            let base_ptr = data.as_mut_ptr();
            _mm256_maskstore_pd(base_ptr, mask1, input.values[0]);
            //let split_index = cmp::min(data.len(), 4);
            let split_index = 4;
            _mm256_maskstore_pd(base_ptr.add(split_index), mask2, input.values[1]);
        }
    }

    fn gather_from_idx(idx: [usize; 8], data: &[f64]) -> Self {
        unsafe {
            let idx_1 = _mm256_loadu_si256(mem::transmute(idx.as_ptr()));
            let v1 = _mm256_i64gather_pd(data.as_ptr(), idx_1, 8);
            let idx_2 = _mm256_loadu_si256(mem::transmute(idx[4..].as_ptr()));
            let v2 = _mm256_i64gather_pd(data.as_ptr(), idx_2, 8);
            Self { values: [v1, v2] }
        }
    }

    fn get_value_at_idx(input: Self, idx: usize) -> f64 {
        unsafe { *input.as_slice().get_unchecked(idx) }
    }

    fn set(value: f64) -> Self {
        unsafe {
            Self {
                values: [_mm256_set1_pd(value), _mm256_set1_pd(value)],
            }
        }
    }

    fn ge(a: Self, b: Self) -> Self::OPMask {
        unsafe {
            let mask_1 = _mm256_cmp_pd(a.values[0], b.values[0], _CMP_GE_OQ);
            let mask_2 = _mm256_cmp_pd(a.values[1], b.values[1], _CMP_GE_OQ);
            Self {
                values: [mask_1, mask_2],
            }
        }
    }

    fn ones_count(mask: Self::OPMask) -> usize {
        unsafe {
            let count1 = _mm256_movemask_pd(mask.values[0]);
            let count2 = _mm256_movemask_pd(mask.values[1]);
            (count1.count_ones() + count2.count_ones()) as usize
        }
    }

    fn not_mask(mask: Self::OPMask) -> Self::OPMask {
        unsafe {
            let all_bit_set = mem::transmute([-1i64, -1, -1, -1]);

            Self {
                values: [
                    _mm256_xor_pd(all_bit_set, mask.values[0]),
                    _mm256_xor_pd(all_bit_set, mask.values[1]),
                ],
            }
        }
    }

    fn reducemin(x: Self) -> f64 {
        unsafe {
            let min_4 = _mm256_min_pd(x.values[0], x.values[1]);
            let v1 = _mm256_extractf128_pd(min_4, 0);
            let v2 = _mm256_extractf128_pd(min_4, 1);
            let min_2 = _mm_min_pd(v1, v2);
            let min_2_rev = _mm_permute_pd(min_2, 0b01);
            let min_result = _mm_min_pd(min_2, min_2_rev);
            mem::transmute::<_, [f64; 2]>(min_result)[0]
        }
    }

    fn reducemax(x: Self) -> f64 {
        unsafe {
            let max_4 = _mm256_max_pd(x.values[0], x.values[1]);
            let v1 = _mm256_extractf128_pd(max_4, 0);
            let v2 = _mm256_extractf128_pd(max_4, 1);
            let max_2 = _mm_max_pd(v1, v2);
            let max_2_rev = _mm_permute_pd(max_2, 0b01);
            let max_result = _mm_max_pd(max_2, max_2_rev);
            mem::transmute::<_, [f64; 2]>(max_result)[0]
        }
    }

    fn mask_compressstoreu(array: &mut [f64], mask: Self::OPMask, data: Self) {
        // get_unchecked call is used to get rid of bound checks
        unsafe {
            let base_ptr = array.as_mut_ptr();
            let bitmask1 = _mm256_movemask_pd(mask.values[0]) as usize;
            let mask1 = _mm256_loadu_si256(mem::transmute(
                COMPRESS_MASK
                    .get_unchecked(bitmask1.count_ones() as usize)
                    .as_ptr(),
            ));
            let v1 = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(
                _mm256_castpd_si256(data.values[0]),
                _mm256_loadu_si256(mem::transmute(
                    COMPRESS_PERMUTATIONS.get_unchecked(bitmask1).as_ptr(),
                )),
            ));
            _mm256_maskstore_pd(base_ptr, mask1, v1);
            let bitmask2 = _mm256_movemask_pd(mask.values[1]) as usize;
            let mask2 = _mm256_loadu_si256(mem::transmute(
                COMPRESS_MASK
                    .get_unchecked(bitmask2.count_ones() as usize)
                    .as_ptr(),
            ));
            let v2 = _mm256_castsi256_pd(_mm256_permutevar8x32_epi32(
                _mm256_castpd_si256(data.values[1]),
                _mm256_loadu_si256(mem::transmute(
                    COMPRESS_PERMUTATIONS.get_unchecked(bitmask2).as_ptr(),
                )),
            ));
            _mm256_maskstore_pd(base_ptr.offset(bitmask1.count_ones() as isize), mask2, v2);
        }
    }
}

impl Bit64Simd<f64> for Avx2F64x2 {
    fn swizzle2_0xaa(a: Self, b: Self) -> Self {
        let v1 = shuffle_256d::<0b1010>(a.values[0], b.values[0]);
        let v2 = shuffle_256d::<0b1010>(a.values[1], b.values[1]);
        Self { values: [v1, v2] }
    }

    fn swizzle2_0xcc(a: Self, b: Self) -> Self {
        unsafe {
            let mask = _mm256_loadu_pd(mem::transmute([0i64, 0, -1, -1].as_ptr()));
            let v1 = blend_256d(a.values[0], b.values[0], mask);
            let v2 = blend_256d(a.values[1], b.values[1], mask);
            Self { values: [v1, v2] }
        }
    }

    fn swizzle2_0xf0(a: Self, b: Self) -> Self {
        Self {
            values: [a.values[0], b.values[1]],
        }
    }

    fn shuffle1_1_1_1(a: Self) -> Self {
        unsafe {
            let v1 = _mm256_permute4x64_pd(a.values[0], SHUFFLE1_1_1_1);
            let v2 = _mm256_permute4x64_pd(a.values[1], SHUFFLE1_1_1_1);
            Self { values: [v1, v2] }
        }
    }

    // 3, 2, 1, 0, 7, 6, 5, 4
    fn network64bit1(a: Self) -> Self {
        unsafe {
            let v1 = _mm256_permute4x64_pd(a.values[0], NETWORK_64BIT_1);
            let v2 = _mm256_permute4x64_pd(a.values[1], NETWORK_64BIT_1);
            Self { values: [v1, v2] }
        }
    }

    // 7, 6, 5, 4, 3, 2, 1, 0
    fn network64bit2(a: Self) -> Self {
        unsafe {
            let v1 = _mm256_permute4x64_pd(a.values[0], NETWORK_64BIT_2);
            let v2 = _mm256_permute4x64_pd(a.values[1], NETWORK_64BIT_2);
            Self { values: [v2, v1] }
        }
    }

    // 2, 3, 0, 1, 6, 7, 4, 5
    fn network64bit3(a: Self) -> Self {
        unsafe {
            let v1 = _mm256_permute4x64_pd(a.values[0], NETWORK_64BIT_3);
            let v2 = _mm256_permute4x64_pd(a.values[1], NETWORK_64BIT_3);
            Self { values: [v1, v2] }
        }
    }

    // 4, 5, 6, 7, 0, 1, 2, 3
    fn network64bit4(a: Self) -> Self {
        Self {
            values: [a.values[1], a.values[0]],
        }
    }
}

#[cfg(test)]
#[cfg(target_feature = "avx2")]
mod test {
    use super::*;
    use crate::bit_64::test::*;

    fn into_array_f64(x: Avx2F64x2) -> [f64; 8] {
        unsafe { mem::transmute(x) }
    }

    trait FromBits {
        fn from_bits(bits: u64) -> Self;
    }

    impl FromBits for f64 {
        fn from_bits(bits: u64) -> Self {
            f64::from_bits(bits)
        }
    }

    fn generate_mask_answer<T, M>(bitmask: usize, values: &[T]) -> (M, [T; 8])
    where
        T: Default + Copy + FromBits,
        M: From<[T; 8]>,
    {
        let mut result = [<T as Default>::default(); 8];
        let mut new_values = [<T as Default>::default(); 8];
        let mut count = 0;
        for i in 0..8 {
            if bitmask & (1 << i) != 0 {
                result[i] = T::from_bits(!0u64);
                new_values[count] = values[i];
                count += 1;
            }
        }
        (M::try_from(result).unwrap(), new_values)
    }

    fn mask_fn(x: u8) -> Avx2F64x2 {
        unsafe { mem::transmute(LOADU_MASK[x.count_ones() as usize]) }
    }

    test_min_max!(f64, Avx2F64x2, into_array_f64);
    test_loadu_storeu!(f64, Avx2F64x2, into_array_f64);
    test_mask_loadu_mask_storeu!(f64, Avx2F64x2, into_array_f64);
    test_get_at_index!(f64, Avx2F64x2);
    test_ge!(f64, Avx2F64x2, unsafe {
        mem::transmute([0i64, -1, 0, -1, 0, -1, 0, -1])
    });
    test_gather!(f64, Avx2F64x2, into_array_f64);
    test_not!(
        f64,
        Avx2F64x2,
        unsafe { mem::transmute([0i64, -1, 0, -1, 0, -1, 0, 0]) },
        unsafe { mem::transmute([-1i64, 0, -1, 0, -1, 0, -1, -1]) }
    );
    test_count_ones!(f64, Avx2F64x2, mask_fn);
    test_reduce_min_max!(f64, Avx2F64x2);
    test_compress_store_u!(f64, Avx2F64x2, Avx2F64x2, generate_mask_answer);
    test_shuffle1_1_1_1!(f64, Avx2F64x2, into_array_f64);
    test_swizzle2_0xaa!(f64, Avx2F64x2, into_array_f64);
    test_swizzle2_0xcc!(f64, Avx2F64x2, into_array_f64);
    test_swizzle2_0xf0!(f64, Avx2F64x2, into_array_f64);
    network64bit1!(f64, Avx2F64x2, into_array_f64);
    network64bit2!(f64, Avx2F64x2, into_array_f64);
    network64bit3!(f64, Avx2F64x2, into_array_f64);
    network64bit4!(f64, Avx2F64x2, into_array_f64);
}
