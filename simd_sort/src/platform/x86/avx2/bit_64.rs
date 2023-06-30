use std::{
    arch::x86_64::{
        __m256i, _mm256_blendv_pd, _mm256_broadcastq_epi64, _mm256_castpd_si256,
        _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _mm256_extracti128_si256,
        _mm256_i64gather_epi64, _mm256_loadu_si256, _mm256_mask_i64gather_epi64,
        _mm256_maskstore_epi64, _mm256_movemask_pd, _mm256_permute4x64_epi64,
        _mm256_permutevar8x32_epi32, _mm256_setr_epi64x, _mm256_shuffle_pd, _mm256_storeu_si256,
        _mm256_xor_si256, _mm_blendv_pd, _mm_castpd_si128, _mm_castsi128_pd, _mm_cmpgt_epi64,
        _mm_extract_epi64, _mm_set1_epi64x, _mm_unpackhi_epi64, _mm_unpacklo_epi64,
    },
    cmp, mem, slice,
};

use crate::{bit_64::Bit64Simd, SimdCompare};

const LOADU_MASK: [[i64; 8]; 9] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0, 0, 0],
    [-1, -1, -1, 0, 0, 0, 0, 0],
    [-1, -1, -1, -1, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1, 0, 0, 0],
    [-1, -1, -1, -1, -1, -1, 0, 0],
    [-1, -1, -1, -1, -1, -1, -1, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1],
];

const V_INDEX_1: [i64; 4] = [0, 1, 2, 3];
const V_INDEX_2: [i64; 4] = [4, 5, 6, 7];

/// [1, 0, 3, 2, 5, 4, 7, 6]
const SHUFFLE1_1_1_1: i32 = 0b10110001;

const COMPRESS_PERMUTATIONS: [[i32; 8]; 16] = [
    [0, 1, 2, 3, 4, 5, 6, 7], //0000
    [0, 1, 2, 3, 4, 5, 6, 7], //0001
    [2, 3, 2, 3, 4, 5, 6, 7], //0010
    [0, 1, 2, 3, 4, 5, 6, 7], //0011
    [4, 5, 2, 3, 4, 5, 6, 7], //0100
    [0, 1, 4, 5, 4, 5, 6, 7], //0101
    [2, 3, 4, 5, 4, 5, 6, 7], //0110
    [0, 1, 2, 3, 4, 5, 6, 7], //0111
    [6, 7, 2, 3, 4, 5, 6, 7], //1000
    [0, 1, 6, 7, 4, 5, 6, 7], //1001
    [2, 3, 6, 7, 4, 5, 6, 7], //1010
    [0, 1, 2, 3, 6, 7, 6, 7], //1011
    [4, 5, 6, 7, 4, 5, 6, 7], //1100
    [0, 1, 4, 5, 6, 7, 6, 7], //1101
    [2, 3, 4, 5, 6, 7, 6, 7], //1110
    [0, 1, 2, 3, 4, 5, 6, 7], //1111
];
const COMPRESS_MASK: [[i64; 4]; 5] = [
    [0, 0, 0, 0],
    [-1, 0, 0, 0],
    [-1, -1, 0, 0],
    [-1, -1, -1, 0],
    [-1, -1, -1, -1],
];

// const COMPRESS_MASK: [[i64; 4]; 5] = [
//     [0, 0, 0, 0],
//     [0, 0, 0, -1],
//     [0, 0, -1, -1],
//     [0, -1, -1, -1],
//     [-1, -1, -1, -1],
// ];

#[inline]
fn blend_256i(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
    unsafe {
        _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(a),
            _mm256_castsi256_pd(b),
            _mm256_castsi256_pd(mask),
        ))
    }
}

#[inline]
fn shuffle_256i<const MASK: i32>(a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        _mm256_castpd_si256(_mm256_shuffle_pd(
            _mm256_castsi256_pd(a),
            _mm256_castsi256_pd(b),
            MASK,
        ))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Avx2I64x2 {
    values: [__m256i; 2],
}

impl PartialEq for Avx2I64x2 {
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

impl SimdCompare<i64, 8> for Avx2I64x2 {
    type OPMask = Self;

    #[inline]
    fn min(a: Self, b: Self) -> Self {
        unsafe {
            let mask1 = _mm256_cmpgt_epi64(a.values[0], b.values[0]);
            let value1 = blend_256i(a.values[0], b.values[0], mask1);
            let mask2 = _mm256_cmpgt_epi64(a.values[1], b.values[1]);
            let value2 = blend_256i(a.values[1], b.values[1], mask2);
            return Self {
                values: [value1, value2],
            };
        }
    }

    #[inline]
    fn max(a: Self, b: Self) -> Self {
        unsafe {
            let mask1 = _mm256_cmpgt_epi64(a.values[0], b.values[0]);
            let value1 = blend_256i(b.values[0], a.values[0], mask1);
            let mask2 = _mm256_cmpgt_epi64(a.values[1], b.values[1]);
            let value2 = blend_256i(b.values[1], a.values[1], mask2);
            return Self {
                values: [value1, value2],
            };
        }
    }

    #[inline]
    fn loadu(data: &[i64]) -> Self {
        unsafe {
            let v1 = _mm256_loadu_si256(mem::transmute(data.as_ptr()));
            let v2 = _mm256_loadu_si256(mem::transmute(data[4..].as_ptr()));
            Self { values: [v1, v2] }
        }
    }

    #[inline]
    fn storeu(input: Self, data: &mut [i64]) {
        unsafe {
            _mm256_storeu_si256(mem::transmute(data.as_ptr()), input.values[0]);
            _mm256_storeu_si256(mem::transmute(data[4..].as_ptr()), input.values[1]);
        }
    }

    fn mask_loadu(data: &[i64]) -> Self {
        Self::from(data)
    }

    fn mask_storeu(input: Self, data: &mut [i64]) {
        unsafe {
            let mask = LOADU_MASK[data.len()];
            let mask1 = _mm256_loadu_si256(mem::transmute(mask.as_ptr()));
            let mask2 = _mm256_loadu_si256(mem::transmute(mask[4..].as_ptr()));
            _mm256_maskstore_epi64(data.as_mut_ptr(), mask1, input.values[0]);
            let split_index = cmp::min(data.len(), 4);
            _mm256_maskstore_epi64(data[split_index..].as_mut_ptr(), mask2, input.values[1]);
        }
    }

    #[inline]
    fn gather_from_idx(idx: [usize; 8], data: &[i64]) -> Self {
        unsafe {
            let idx_1 = _mm256_loadu_si256(mem::transmute(idx.as_ptr()));
            let v1 = _mm256_i64gather_epi64(data.as_ptr(), idx_1, 8);
            let idx_2 = _mm256_loadu_si256(mem::transmute(idx[4..].as_ptr()));
            let v2 = _mm256_i64gather_epi64(data.as_ptr(), idx_2, 8);
            Self { values: [v1, v2] }
        }
    }

    #[inline]
    fn get_value_at_idx(input: Self, idx: usize) -> i64 {
        input.as_slice()[idx]
    }

    #[inline]
    fn set(value: i64) -> Self {
        unsafe {
            let braocast_128 = _mm_set1_epi64x(value);
            let v1 = _mm256_broadcastq_epi64(braocast_128);
            let v2 = _mm256_broadcastq_epi64(braocast_128);
            Self { values: [v1, v2] }
        }
    }

    #[inline]
    fn ge(a: Self, b: Self) -> Self::OPMask {
        unsafe {
            let gt_mask1 = _mm256_cmpgt_epi64(a.values[0], b.values[0]);
            let gt_mask2 = _mm256_cmpgt_epi64(a.values[1], b.values[1]);

            let eq_mask1 = _mm256_cmpeq_epi64(a.values[0], b.values[0]);
            let eq_mask2 = _mm256_cmpeq_epi64(a.values[1], b.values[1]);

            let mask1 = _mm256_xor_si256(gt_mask1, eq_mask1);
            let mask2 = _mm256_xor_si256(gt_mask2, eq_mask2);

            return Self {
                values: [mask1, mask2],
            };
        }
    }

    #[inline]
    fn ones_count(mask: Self::OPMask) -> usize {
        unsafe {
            let count1 = _mm256_movemask_pd(_mm256_castsi256_pd(mask.values[0]));
            let count2 = _mm256_movemask_pd(_mm256_castsi256_pd(mask.values[1]));
            (count1.count_ones() + count2.count_ones()) as usize
        }
    }

    #[inline]
    fn not_mask(mask: Self::OPMask) -> Self::OPMask {
        unsafe {
            let all_bit_set = _mm256_cmpeq_epi64(mask.values[0], mask.values[0]);

            Self {
                values: [
                    _mm256_xor_si256(all_bit_set, mask.values[0]),
                    _mm256_xor_si256(all_bit_set, mask.values[1]),
                ],
            }
        }
    }

    #[inline]
    fn reducemin(x: Self) -> i64 {
        unsafe {
            let gt_mask_4 = _mm256_cmpgt_epi64(x.values[0], x.values[1]);
            let max_4 = blend_256i(x.values[0], x.values[1], gt_mask_4);
            let max_2_1 = _mm256_extracti128_si256(max_4, 0);
            let max_2_2 = _mm256_extracti128_si256(max_4, 1);
            let gt_mask_2 = _mm_cmpgt_epi64(max_2_1, max_2_2);
            let max_2 = _mm_castpd_si128(_mm_blendv_pd(
                _mm_castsi128_pd(max_2_1),
                _mm_castsi128_pd(max_2_2),
                _mm_castsi128_pd(gt_mask_2),
            ));
            let lo = _mm_unpacklo_epi64(max_2, max_2);
            let hi = _mm_unpackhi_epi64(max_2, max_2);
            let gt_mask = _mm_cmpgt_epi64(lo, hi);
            let max_final = _mm_castpd_si128(_mm_blendv_pd(
                _mm_castsi128_pd(lo),
                _mm_castsi128_pd(hi),
                _mm_castsi128_pd(gt_mask),
            ));
            _mm_extract_epi64(max_final, 0)
        }
    }

    #[inline]
    fn reducemax(x: Self) -> i64 {
        unsafe {
            let gt_mask_4 = _mm256_cmpgt_epi64(x.values[0], x.values[1]);
            let max_4 = blend_256i(x.values[1], x.values[0], gt_mask_4);
            let max_2_1 = _mm256_extracti128_si256(max_4, 0);
            let max_2_2 = _mm256_extracti128_si256(max_4, 1);
            let gt_mask_2 = _mm_cmpgt_epi64(max_2_1, max_2_2);
            let max_2 = _mm_castpd_si128(_mm_blendv_pd(
                _mm_castsi128_pd(max_2_2),
                _mm_castsi128_pd(max_2_1),
                _mm_castsi128_pd(gt_mask_2),
            ));
            let lo = _mm_unpacklo_epi64(max_2, max_2);
            let hi = _mm_unpackhi_epi64(max_2, max_2);
            let gt_mask = _mm_cmpgt_epi64(lo, hi);
            let max_final = _mm_castpd_si128(_mm_blendv_pd(
                _mm_castsi128_pd(hi),
                _mm_castsi128_pd(lo),
                _mm_castsi128_pd(gt_mask),
            ));
            _mm_extract_epi64(max_final, 0)
        }
    }

    #[inline]
    fn mask_compressstoreu(array: &mut [i64], mask: Self::OPMask, data: Self) {
        unsafe {
            let bitmask1 = _mm256_movemask_pd(_mm256_castsi256_pd(mask.values[0])) as usize;
            let mask1 = _mm256_loadu_si256(mem::transmute(
                COMPRESS_MASK[bitmask1.count_ones() as usize].as_ptr(),
            ));
            //dbg!(format!("{:?}", mask1));
            //dbg!(format!("{:?}", data.values[0]));
            let v1 = _mm256_permutevar8x32_epi32(
                data.values[0],
                _mm256_loadu_si256(mem::transmute(COMPRESS_PERMUTATIONS[bitmask1].as_ptr())),
            );
            //dbg!(format!("{:?}", v1));
            _mm256_maskstore_epi64(array.as_mut_ptr(), mask1, v1);
            let bitmask2 = _mm256_movemask_pd(_mm256_castsi256_pd(mask.values[1])) as usize;
            let mask2 = _mm256_loadu_si256(mem::transmute(
                COMPRESS_MASK[bitmask2.count_ones() as usize].as_ptr(),
            ));
            //dbg!(format!("{:?}", mask1));
            //dbg!(format!("{:?}", data.values[1]));
            let v2 = _mm256_permutevar8x32_epi32(
                data.values[1],
                _mm256_loadu_si256(mem::transmute(COMPRESS_PERMUTATIONS[bitmask2].as_ptr())),
            );
            //dbg!(format!("{:?}", v2));
            _mm256_maskstore_epi64(
                array[bitmask1.count_ones() as usize..].as_mut_ptr(),
                mask2,
                v2,
            );
        }
    }
}

impl Avx2I64x2 {
    fn as_slice(&self) -> &[i64] {
        unsafe { slice::from_raw_parts(mem::transmute(self.values.as_ptr()), 8) }
    }
}

impl From<[i64; 8]> for Avx2I64x2 {
    fn from(v: [i64; 8]) -> Self {
        unsafe {
            Self {
                values: [
                    _mm256_setr_epi64x(v[0], v[1], v[2], v[3]),
                    _mm256_setr_epi64x(v[4], v[5], v[6], v[7]),
                ],
            }
        }
    }
}

impl From<&[i64]> for Avx2I64x2 {
    fn from(v: &[i64]) -> Self {
        unsafe {
            let mask = LOADU_MASK[v.len()].as_ptr();
            let mask1 = _mm256_loadu_si256(mem::transmute(mask));
            let mask2 = _mm256_loadu_si256(mem::transmute(mask.offset(4)));
            let indices1 = _mm256_loadu_si256(mem::transmute(V_INDEX_1.as_ptr()));
            let indices2 = _mm256_loadu_si256(mem::transmute(V_INDEX_2.as_ptr()));
            let max_values = _mm256_broadcastq_epi64(_mm_set1_epi64x(i64::MAX));
            let v1 = _mm256_mask_i64gather_epi64(max_values, v.as_ptr(), indices1, mask1, 8);
            let v2 = _mm256_mask_i64gather_epi64(max_values, v.as_ptr(), indices2, mask2, 8);
            Self { values: [v1, v2] }
        }
    }
}

impl Bit64Simd<i64> for Avx2I64x2 {
    fn swizzle2_0xaa(a: Self, b: Self) -> Self {
        let v1 = shuffle_256i::<0b1010>(a.values[0], b.values[0]);
        let v2 = shuffle_256i::<0b1010>(a.values[1], b.values[1]);
        Self { values: [v1, v2] }
    }

    fn swizzle2_0xcc(a: Self, b: Self) -> Self {
        unsafe {
            let mask = _mm256_loadu_si256(mem::transmute([0i64, 0, -1, -1].as_ptr()));
            let v1 = blend_256i(a.values[0], b.values[0], mask);
            let v2 = blend_256i(a.values[1], b.values[1], mask);
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
            let v1 = _mm256_permute4x64_epi64(a.values[0], SHUFFLE1_1_1_1);
            let v2 = _mm256_permute4x64_epi64(a.values[1], SHUFFLE1_1_1_1);
            Self { values: [v1, v2] }
        }
    }

    // 3, 2, 1, 0, 7, 6, 5, 4
    fn network64bit1(a: Self) -> Self {
        unsafe {
            let v1 = _mm256_permute4x64_epi64(a.values[0], 0b00011011);
            let v2 = _mm256_permute4x64_epi64(a.values[1], 0b00011011);
            Self { values: [v1, v2] }
        }
    }

    // 7, 6, 5, 4, 3, 2, 1, 0
    fn network64bit2(a: Self) -> Self {
        unsafe {
            let v1 = _mm256_permute4x64_epi64(a.values[0], 0b00011011);
            let v2 = _mm256_permute4x64_epi64(a.values[1], 0b00011011);
            Self { values: [v2, v1] }
        }
    }

    // 2, 3, 0, 1, 6, 7, 4, 5
    fn network64bit3(a: Self) -> Self {
        unsafe {
            let v1 = _mm256_permute4x64_epi64(a.values[0], 0b01001110);
            let v2 = _mm256_permute4x64_epi64(a.values[1], 0b01001110);
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
#[cfg(target_feature="avx2")]
mod test {
    use crate::bit_64::test::*;
    use std::fmt::Debug;
    use std::ops::Not;

    use super::*;

    fn into_array_i64(x: Avx2I64x2) -> [i64; 8] {
        unsafe {
            mem::transmute(x)
        }
    }

    fn generate_mask_answer<T, M>(bitmask: usize, values: &[T]) -> (M, [T; 8]) 
    where
        T: TryFrom<usize> + Default + Copy +Not<Output = T>,
        <T as TryFrom<usize>>::Error: Debug,
        M: From<[T;8]>
    {
        let mut result = [<T as Default>::default(); 8];
        let mut new_values = [<T as Default>::default(); 8];
        let mut count = 0;
        for i in 0..8 {
            if bitmask & (1 << i) != 0 {
                result[i] = !<T as Default>::default();
                new_values[count] = values[i];
                count += 1;
            }
        }
        (M::try_from(result).unwrap(), new_values)
    }

    fn mask_fn(x: u8) -> Avx2I64x2 {
        Avx2I64x2::from(LOADU_MASK[x.count_ones() as usize])
    }

    test_min_max!(i64, Avx2I64x2, into_array_i64);
    test_loadu_storeu!(i64, Avx2I64x2, into_array_i64);
    test_mask_loadu_mask_storeu!(i64, Avx2I64x2, into_array_i64);
    test_get_at_index!(i64, Avx2I64x2);
    test_ge!(i64, Avx2I64x2, Avx2I64x2::from([0, -1, 0, -1, 0, -1, 0, -1]));
    test_gather!(i64, Avx2I64x2, into_array_i64);
    test_not!(i64, Avx2I64x2, Avx2I64x2::from([0, -1, 0, -1, 0, -1, 0, 0]), Avx2I64x2::from([-1, 0, -1, 0, -1, 0, -1, -1]));
    test_count_ones!(i64, Avx2I64x2, mask_fn);
    test_reduce_min_max!(i64, Avx2I64x2);
    test_compress_store_u!(i64, Avx2I64x2, Avx2I64x2, generate_mask_answer);
    test_shuffle1_1_1_1!(i64, Avx2I64x2, into_array_i64);
    test_swizzle2_0xaa!(i64, Avx2I64x2, into_array_i64);
    test_swizzle2_0xcc!(i64, Avx2I64x2, into_array_i64);
    test_swizzle2_0xf0!(i64, Avx2I64x2, into_array_i64);
    network64bit1!(i64, Avx2I64x2, into_array_i64);
    network64bit2!(i64, Avx2I64x2, into_array_i64);
    network64bit3!(i64, Avx2I64x2, into_array_i64);
    network64bit4!(i64, Avx2I64x2, into_array_i64);
}
