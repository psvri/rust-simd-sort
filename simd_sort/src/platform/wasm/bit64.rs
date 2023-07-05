use std::{
    arch::wasm32::{
        f64x2, i64x2, i64x2_bitmask, i64x2_extract_lane, i64x2_ge, i64x2_gt, i64x2_lt,
        i64x2_shuffle, u64x2, v128, v128_bitselect, v128_load, v128_store,
    },
    mem::{self},
    ptr,
};

use crate::{bit_64::Bit64Simd, SimdCompare};

#[derive(Debug, Clone, Copy)]
pub struct Wasmi64x8 {
    values: [v128; 4],
}

impl Wasmi64x8 {
    fn min_x2(a: v128, b: v128) -> v128 {
        v128_bitselect(a, b, i64x2_lt(a, b))
    }

    fn max_x2(a: v128, b: v128) -> v128 {
        v128_bitselect(a, b, i64x2_gt(a, b))
    }
}

impl SimdCompare<i64, 8> for Wasmi64x8 {
    type OPMask = u8;

    fn min(a: Self, b: Self) -> Self {
        Wasmi64x8 {
            values: [
                Self::min_x2(a.values[0], b.values[0]),
                Self::min_x2(a.values[1], b.values[1]),
                Self::min_x2(a.values[2], b.values[2]),
                Self::min_x2(a.values[3], b.values[3]),
            ],
        }
    }

    fn max(a: Self, b: Self) -> Self {
        Wasmi64x8 {
            values: [
                Self::max_x2(a.values[0], b.values[0]),
                Self::max_x2(a.values[1], b.values[1]),
                Self::max_x2(a.values[2], b.values[2]),
                Self::max_x2(a.values[3], b.values[3]),
            ],
        }
    }

    fn loadu(data: &[i64]) -> Self {
        unsafe {
            let base_ptr = data.as_ptr();

            Wasmi64x8 {
                values: [
                    v128_load(mem::transmute(base_ptr)),
                    v128_load(mem::transmute(base_ptr.offset(2))),
                    v128_load(mem::transmute(base_ptr.offset(4))),
                    v128_load(mem::transmute(base_ptr.offset(6))),
                ],
            }
        }
    }

    fn storeu(input: Self, data: &mut [i64]) {
        unsafe {
            let base_ptr = data.as_mut_ptr();

            v128_store(mem::transmute(base_ptr), input.values[0]);
            v128_store(mem::transmute(base_ptr.offset(2)), input.values[1]);
            v128_store(mem::transmute(base_ptr.offset(4)), input.values[2]);
            v128_store(mem::transmute(base_ptr.offset(6)), input.values[3]);
        }
    }

    fn mask_loadu(data: &[i64]) -> Self {
        let mut max = [i64::MAX; 8];
        max[..data.len()].copy_from_slice(data);
        unsafe {
            ptr::copy(data.as_ptr(), max.as_mut_ptr(), data.len());
            Wasmi64x8 {
                values: mem::transmute(max),
            }
        }
    }

    fn mask_storeu(input: Self, data: &mut [i64]) {
        unsafe {
            ptr::copy(
                mem::transmute(input.values.as_ptr()),
                data.as_mut_ptr(),
                data.len(),
            );
        }
    }

    fn gather_from_idx(idx: [usize; 8], data: &[i64]) -> Self {
        unsafe {
            let base_ptr = data.as_ptr();
            Wasmi64x8 {
                values: [
                    i64x2(
                        *base_ptr.offset(idx[0] as isize),
                        *base_ptr.offset(idx[1] as isize),
                    ),
                    i64x2(
                        *base_ptr.offset(idx[2] as isize),
                        *base_ptr.offset(idx[3] as isize),
                    ),
                    i64x2(
                        *base_ptr.offset(idx[4] as isize),
                        *base_ptr.offset(idx[5] as isize),
                    ),
                    i64x2(
                        *base_ptr.offset(idx[6] as isize),
                        *base_ptr.offset(idx[7] as isize),
                    ),
                ],
            }
        }
    }

    fn get_value_at_idx(input: Self, idx: usize) -> i64 {
        unsafe {
            let base_ptr: *const i64 = mem::transmute(&input);
            *base_ptr.offset(idx as isize)
        }
    }

    fn set(value: i64) -> Self {
        Wasmi64x8 {
            values: [
                i64x2(value, value),
                i64x2(value, value),
                i64x2(value, value),
                i64x2(value, value),
            ],
        }
    }

    fn ge(a: Self, b: Self) -> Self::OPMask {
        i64x2_bitmask(i64x2_ge(a.values[0], b.values[0]))
            | i64x2_bitmask(i64x2_ge(a.values[1], b.values[1])) << 2
            | i64x2_bitmask(i64x2_ge(a.values[2], b.values[2])) << 4
            | i64x2_bitmask(i64x2_ge(a.values[3], b.values[3])) << 6
    }

    fn ones_count(mask: Self::OPMask) -> usize {
        mask.count_ones() as usize
    }

    fn not_mask(mask: Self::OPMask) -> Self::OPMask {
        !mask
    }

    fn reducemin(x: Self) -> i64 {
        let m1 = Self::min_x2(x.values[0], x.values[1]);
        let m2 = Self::min_x2(x.values[2], x.values[3]);
        let m2 = Self::min_x2(m1, m2);
        let m2_shuffle = i64x2_shuffle::<1, 0>(m2, m2);
        let m2 = Self::min_x2(m2, m2_shuffle);
        i64x2_extract_lane::<0>(m2)
    }

    fn reducemax(x: Self) -> i64 {
        let m1 = Self::max_x2(x.values[0], x.values[1]);
        let m2 = Self::max_x2(x.values[2], x.values[3]);
        let m2 = Self::max_x2(m1, m2);
        let m2_shuffle = i64x2_shuffle::<1, 0>(m2, m2);
        let m2 = Self::max_x2(m2, m2_shuffle);
        i64x2_extract_lane::<0>(m2)
    }

    fn mask_compressstoreu(array: &mut [i64], mask: Self::OPMask, data: Self) {
        unsafe {
            let mut temp_mask = mask;
            let mut base_ptr = array.as_mut_ptr();
            let value_ptr: *const i64 = mem::transmute(&data);

            for i in 0..8 {
                if temp_mask & 1 == 1 {
                    *base_ptr = *value_ptr.offset(i);
                    base_ptr = base_ptr.offset(1);
                }
                temp_mask = temp_mask >> 1;
            }
        }
    }
}

impl Bit64Simd<i64> for Wasmi64x8 {
    fn swizzle2_0xaa(a: Self, b: Self) -> Self {
        Self {
            values: [
                i64x2_shuffle::<0, 3>(a.values[0], b.values[0]),
                i64x2_shuffle::<0, 3>(a.values[1], b.values[1]),
                i64x2_shuffle::<0, 3>(a.values[2], b.values[2]),
                i64x2_shuffle::<0, 3>(a.values[3], b.values[3]),
            ],
        }
    }

    fn swizzle2_0xcc(a: Self, b: Self) -> Self {
        Self {
            values: [a.values[0], b.values[1], a.values[2], b.values[3]],
        }
    }

    fn swizzle2_0xf0(a: Self, b: Self) -> Self {
        Self {
            values: [a.values[0], a.values[1], b.values[2], b.values[3]],
        }
    }

    fn shuffle1_1_1_1(a: Self) -> Self {
        Self {
            values: [
                i64x2_shuffle::<1, 0>(a.values[0], a.values[0]),
                i64x2_shuffle::<1, 0>(a.values[1], a.values[1]),
                i64x2_shuffle::<1, 0>(a.values[2], a.values[2]),
                i64x2_shuffle::<1, 0>(a.values[3], a.values[3]),
            ],
        }
    }

    fn network64bit1(a: Self) -> Self {
        Self {
            values: [
                i64x2_shuffle::<1, 0>(a.values[1], a.values[1]),
                i64x2_shuffle::<1, 0>(a.values[0], a.values[0]),
                i64x2_shuffle::<1, 0>(a.values[3], a.values[3]),
                i64x2_shuffle::<1, 0>(a.values[2], a.values[2]),
            ],
        }
    }

    fn network64bit2(a: Self) -> Self {
        Self {
            values: [
                i64x2_shuffle::<1, 0>(a.values[3], a.values[3]),
                i64x2_shuffle::<1, 0>(a.values[2], a.values[2]),
                i64x2_shuffle::<1, 0>(a.values[1], a.values[1]),
                i64x2_shuffle::<1, 0>(a.values[0], a.values[0]),
            ],
        }
    }

    fn network64bit3(a: Self) -> Self {
        Self {
            values: [a.values[1], a.values[0], a.values[3], a.values[2]],
        }
    }

    fn network64bit4(a: Self) -> Self {
        Self {
            values: [a.values[2], a.values[3], a.values[0], a.values[1]],
        }
    }
}

#[cfg(test)]
mod test {
    use crate::bit_64::test::*;
    use std::fmt::Debug;

    use super::*;

    fn into_array_i64(x: Wasmi64x8) -> [i64; 8] {
        [
            i64x2_extract_lane::<0>(x.values[0]),
            i64x2_extract_lane::<1>(x.values[0]),
            i64x2_extract_lane::<0>(x.values[1]),
            i64x2_extract_lane::<1>(x.values[1]),
            i64x2_extract_lane::<0>(x.values[2]),
            i64x2_extract_lane::<1>(x.values[2]),
            i64x2_extract_lane::<0>(x.values[3]),
            i64x2_extract_lane::<1>(x.values[3]),
        ]
    }

    fn generate_mask_answer<T, M>(bitmask: usize, values: &[T]) -> (M, [T; 8])
    where
        T: TryFrom<usize> + Default + Copy,
        <T as TryFrom<usize>>::Error: Debug,
        M: TryFrom<usize>,
        <M as TryFrom<usize>>::Error: Debug,
    {
        let mut new_values = [<T as Default>::default(); 8];
        let mut count = 0;
        for i in 0..8 {
            if bitmask & (1 << i) != 0 {
                new_values[count] = values[i];
                count += 1;
            }
        }
        (bitmask.try_into().unwrap(), new_values)
    }

    fn mask_fn(x: u8) -> u8 {
        x
    }

    test_min_max!(i64, Wasmi64x8, into_array_i64);
    test_loadu_storeu!(i64, Wasmi64x8, into_array_i64);
    test_mask_loadu_mask_storeu!(i64, Wasmi64x8, into_array_i64);
    test_get_at_index!(i64, Wasmi64x8);
    test_ge!(i64, Wasmi64x8, 0b10101010);
    test_gather!(i64, Wasmi64x8, into_array_i64);
    test_not!(i64, Wasmi64x8, 5, !5);
    test_count_ones!(i64, Wasmi64x8, mask_fn);
    test_reduce_min_max!(i64, Wasmi64x8);
    test_compress_store_u!(i64, Wasmi64x8, u8, generate_mask_answer);
    test_shuffle1_1_1_1!(i64, Wasmi64x8, into_array_i64);
    test_swizzle2_0xaa!(i64, Wasmi64x8, into_array_i64);
    test_swizzle2_0xcc!(i64, Wasmi64x8, into_array_i64);
    test_swizzle2_0xf0!(i64, Wasmi64x8, into_array_i64);
    network64bit1!(i64, Wasmi64x8, into_array_i64);
    network64bit2!(i64, Wasmi64x8, into_array_i64);
    network64bit3!(i64, Wasmi64x8, into_array_i64);
    network64bit4!(i64, Wasmi64x8, into_array_i64);
}
