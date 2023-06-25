use core::slice;
use std::simd::{
    i64x8, u64x8, usizex8, Simd, SimdInt, SimdOrd, SimdPartialEq, SimdPartialOrd, SimdUint,
    Swizzle, Swizzle2, ToBitMask,
    Which::{self, *},
};

use crate::{
    bit_64::{Bit64Element, Bit64Simd},
    SimdCompare, SimdSortable,
};

//   ZMM                  7, 6, 5, 4, 3, 2, 1, 0
//#define NETWORK_64BIT_1 4, 5, 6, 7, 0, 1, 2, 3
struct Network64bit1;

impl Swizzle<8, 8> for Network64bit1 {
    const INDEX: [usize; 8] = [3, 2, 1, 0, 7, 6, 5, 4];
}

//   ZMM                  7, 6, 5, 4, 3, 2, 1, 0
//#define NETWORK_64BIT_2 0, 1, 2, 3, 4, 5, 6, 7
struct Network64bit2;

impl Swizzle<8, 8> for Network64bit2 {
    const INDEX: [usize; 8] = [7, 6, 5, 4, 3, 2, 1, 0];
}

//   ZMM                  7, 6, 5, 4, 3, 2, 1, 0
//#define NETWORK_64BIT_3 5, 4, 7, 6, 1, 0, 3, 2
struct Network64bit3;

impl Swizzle<8, 8> for Network64bit3 {
    const INDEX: [usize; 8] = [2, 3, 0, 1, 6, 7, 4, 5];
}

//   ZMM                  7, 6, 5, 4, 3, 2, 1, 0
//#define NETWORK_64BIT_4 3, 2, 1, 0, 7, 6, 5, 4
struct Network64bit4;

impl Swizzle<8, 8> for Network64bit4 {
    const INDEX: [usize; 8] = [4, 5, 6, 7, 0, 1, 2, 3];
}

//SHUFFLE_MASK(1, 1, 1, 1)
struct Shuffle1_1_1_1;

impl Swizzle<8, 8> for Shuffle1_1_1_1 {
    const INDEX: [usize; 8] = [1, 0, 3, 2, 5, 4, 7, 6];
}

//ZMM 76543210
//    10101010 , 0 -> first, 1 -> second
struct Swizzle2_0xAA;

impl Swizzle2<8, 8> for Swizzle2_0xAA {
    const INDEX: [Which; 8] = [
        First(0),
        Second(1),
        First(2),
        Second(3),
        First(4),
        Second(5),
        First(6),
        Second(7),
    ];
}

//ZMM 76543210
//    11001100, 0 -> first, 1 -> second
struct Swizzle2_0xCC;

impl Swizzle2<8, 8> for Swizzle2_0xCC {
    const INDEX: [Which; 8] = [
        First(0),
        First(1),
        Second(2),
        Second(3),
        First(4),
        First(5),
        Second(6),
        Second(7),
    ];
}

//ZMM 76543210
//    11110000, 0 -> first, 1 -> second
struct Swizzle2_0xF0;

impl Swizzle2<8, 8> for Swizzle2_0xF0 {
    const INDEX: [Which; 8] = [
        First(0),
        First(1),
        First(2),
        First(3),
        Second(4),
        Second(5),
        Second(6),
        Second(7),
    ];
}

impl SimdCompare<u64, 8> for u64x8 {
    type OPMask = <Simd<u64, 8> as SimdPartialEq>::Mask;

    #[inline]
    fn min(a: Self, b: Self) -> Self {
        a.simd_min(b)
    }

    #[inline]
    fn max(a: Self, b: Self) -> Self {
        a.simd_max(b)
    }

    #[inline]
    fn mask_mov(a: Self, b: Self, mask: fn(Self, Self) -> Self) -> Self {
        mask(a, b)
    }

    #[inline]
    fn shuffle(a: Self, mask: fn(Self) -> Self) -> Self {
        mask(a)
    }

    #[inline]
    fn loadu(data: &[u64]) -> Self {
        let mut values = [0; 8];
        values.copy_from_slice(unsafe { slice::from_raw_parts(data.as_ptr(), 8) });
        Self::from_array(values)
    }

    #[inline]
    fn storeu(input: Self, output: &mut [u64]) {
        unsafe {
            slice::from_raw_parts_mut(output.as_mut_ptr(), 8).copy_from_slice(input.as_array())
        }
    }

    #[inline]
    fn mask_loadu(data: &[u64]) -> Self {
        let idxs = usizex8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
        let max_values = u64x8::splat(u64::MAX_VALUE);
        u64x8::gather_or(data, idxs, max_values)
    }

    #[inline]
    fn mask_storeu(input: Self, data: &mut [u64]) {
        let idxs = usizex8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
        u64x8::scatter(input, data, idxs);
    }

    #[inline]
    fn gather_from_idx(idx: [usize; 8], data: &[u64]) -> Self {
        let idxs = usizex8::from_array(idx);
        let max_values = u64x8::splat(u64::MAX_VALUE);
        u64x8::gather_or(data, idxs, max_values)
    }

    #[inline]
    fn get_value_at_idx(input: Self, idx: usize) -> u64 {
        input[idx]
    }

    #[inline]
    fn set(value: u64) -> Self {
        u64x8::splat(value)
    }

    #[inline]
    fn ge(a: Self, b: Self) -> Self::OPMask {
        //Self::OPMask::from_int(Network64bit2::swizzle(a.simd_ge(b).to_int()))
        a.simd_ge(b)
    }

    #[inline]
    fn ones_count(mask: Self::OPMask) -> usize {
        mask.to_bitmask().count_ones() as usize
    }

    #[inline]
    fn not_mask(mask: Self::OPMask) -> Self::OPMask {
        !mask
    }

    #[inline]
    fn reducemin(data: Self) -> u64 {
        data.reduce_min()
    }

    #[inline]
    fn reducemax(data: Self) -> u64 {
        data.reduce_max()
    }

    #[inline]
    fn mask_compressstoreu(array: &mut [u64], mask: Self::OPMask, vals: Self) {
        let mut ptr = 0;

        let count = mask.to_array();

        for (idx, i) in count.iter().enumerate() {
            if *i {
                array[ptr] = vals[idx];
                ptr += 1;
            }
        }
    }
}

impl SimdCompare<i64, 8> for i64x8 {
    type OPMask = <Simd<i64, 8> as SimdPartialEq>::Mask;

    #[inline]
    fn min(a: Self, b: Self) -> Self {
        a.simd_min(b)
    }

    #[inline]
    fn max(a: Self, b: Self) -> Self {
        a.simd_max(b)
    }

    #[inline]
    fn mask_mov(a: Self, b: Self, mask: fn(Self, Self) -> Self) -> Self {
        mask(a, b)
    }

    #[inline]
    fn shuffle(a: Self, mask: fn(Self) -> Self) -> Self {
        mask(a)
    }

    #[inline]
    fn loadu(data: &[i64]) -> Self {
        let mut values = [0; 8];
        values.copy_from_slice(unsafe { slice::from_raw_parts(data.as_ptr(), 8) });
        Self::from_array(values)
    }

    #[inline]
    fn storeu(input: Self, output: &mut [i64]) {
        unsafe {
            slice::from_raw_parts_mut(output.as_mut_ptr(), 8).copy_from_slice(input.as_array())
        }
    }

    #[inline]
    fn mask_loadu(data: &[i64]) -> Self {
        let idxs = usizex8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
        let max_values = i64x8::splat(i64::MAX_VALUE);
        i64x8::gather_or(data, idxs, max_values)
    }

    #[inline]
    fn mask_storeu(input: Self, output: &mut [i64]) {
        let idxs = usizex8::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
        i64x8::scatter(input, output, idxs);
    }

    #[inline]
    fn gather_from_idx(idx: [usize; 8], data: &[i64]) -> Self {
        let idxs = usizex8::from_array(idx);
        let max_values = i64x8::splat(i64::MAX_VALUE);
        i64x8::gather_or(data, idxs, max_values)
    }

    #[inline]
    fn get_value_at_idx(input: Self, idx: usize) -> i64 {
        input[idx]
    }

    #[inline]
    fn set(value: i64) -> Self {
        i64x8::splat(value)
    }

    #[inline]
    fn ge(a: Self, b: Self) -> Self::OPMask {
        a.simd_ge(b)
    }

    #[inline]
    fn ones_count(mask: Self::OPMask) -> usize {
        mask.to_bitmask().count_ones() as usize
    }

    #[inline]
    fn not_mask(mask: Self::OPMask) -> Self::OPMask {
        !mask
    }

    #[inline]
    fn reducemin(data: Self) -> i64 {
        data.reduce_min()
    }

    #[inline]
    fn reducemax(data: Self) -> i64 {
        data.reduce_max()
    }

    #[inline]
    fn mask_compressstoreu(array: &mut [i64], mask: Self::OPMask, vals: Self) {
        let mut ptr = 0;

        let count = mask.to_array();

        for (idx, i) in count.iter().enumerate() {
            if *i {
                array[ptr] = vals[idx];
                ptr += 1;
            }
        }
    }
}

impl<T: Bit64Element> Bit64Simd<T> for Simd<T, 8> {
    fn swizzle2_0xaa(a: Self, b: Self) -> Self {
        Swizzle2_0xAA::swizzle2(a, b)
    }

    fn swizzle2_0xcc(a: Self, b: Self) -> Self {
        Swizzle2_0xCC::swizzle2(a, b)
    }

    fn swizzle2_0xf0(a: Self, b: Self) -> Self {
        Swizzle2_0xF0::swizzle2(a, b)
    }

    fn shuffle1_1_1_1(a: Self) -> Self {
        Shuffle1_1_1_1::swizzle(a)
    }

    fn network64bit1(a: Self) -> Self {
        Network64bit1::swizzle(a)
    }

    fn network64bit2(a: Self) -> Self {
        Network64bit2::swizzle(a)
    }

    fn network64bit3(a: Self) -> Self {
        Network64bit3::swizzle(a)
    }

    fn network64bit4(a: Self) -> Self {
        Network64bit4::swizzle(a)
    }
}

#[cfg(test)]
mod tests {
    use crate::bit_64::test::*;
    use std::simd::*;

    use super::*;

    fn into_array_i64(x: i64x8) -> [i64; 8] {
        x.into()
    }

    fn into_array_u64(x: u64x8) -> [u64; 8] {
        x.into()
    }

    fn mask_fn<T: MaskElement>(x: u8) -> Mask<T, 8> {
        Mask::<T, 8>::from_bitmask(x)
    }

    fn generate_mask_answer<T, M>(bitmask: usize, values: &[T]) -> (M, [T; 8])
    where
        T: TryFrom<usize> + Default + Copy,
        M: ToBitMask<BitMask = u8>,
    {
        let mut new_values = [<T as Default>::default(); 8];
        let mut count = 0;
        for i in 0..8 {
            if bitmask & (1 << i) != 0 {
                new_values[count] = values[i];
                count += 1;
            }
        }
        (M::from_bitmask(bitmask as u8), new_values)
    }

    type I64Mask = Mask<i64, 8>;

    // test i64x8
    test_min_max!(i64, i64x8, into_array_i64);
    test_loadu_storeu!(i64, i64x8, into_array_i64);
    test_mask_loadu_mask_storeu!(i64, i64x8, into_array_i64);
    test_get_at_index!(i64, i64x8);
    test_ge!(
        i64,
        i64x8,
        Mask::from([false, true, false, true, false, true, false, true])
    );
    test_gather!(i64, i64x8, into_array_i64);
    test_not!(
        i64,
        i64x8,
        Mask::from([false, true, false, true, false, true, false, false]),
        Mask::from([true, false, true, false, true, false, true, true])
    );
    test_count_ones!(i64, i64x8, mask_fn);
    test_reduce_min_max!(i64, i64x8);
    test_compress_store_u!(i64, i64x8, I64Mask, generate_mask_answer);
    test_shuffle1_1_1_1!(i64, i64x8, into_array_i64);
    test_swizzle2_0xaa!(i64, i64x8, into_array_i64);
    test_swizzle2_0xcc!(i64, i64x8, into_array_i64);
    test_swizzle2_0xf0!(i64, i64x8, into_array_i64);
    network64bit1!(i64, i64x8, into_array_i64);
    network64bit2!(i64, i64x8, into_array_i64);
    network64bit3!(i64, i64x8, into_array_i64);
    network64bit4!(i64, i64x8, into_array_i64);

    // test u64x8
    test_min_max!(u64, u64x8, into_array_u64);
    test_loadu_storeu!(u64, u64x8, into_array_u64);
    test_mask_loadu_mask_storeu!(u64, u64x8, into_array_u64);
    test_get_at_index!(u64, u64x8);
    test_ge!(
        u64,
        u64x8,
        Mask::from([false, true, false, true, false, true, false, true])
    );
    test_gather!(u64, u64x8, into_array_u64);
    test_not!(
        u64,
        u64x8,
        Mask::from([false, true, false, true, false, true, false, false]),
        Mask::from([true, false, true, false, true, false, true, true])
    );
    test_count_ones!(u64, u64x8, mask_fn);
    test_reduce_min_max!(u64, u64x8);
    test_compress_store_u!(u64, u64x8, I64Mask, generate_mask_answer);
    test_shuffle1_1_1_1!(u64, u64x8, into_array_u64);
    test_swizzle2_0xaa!(u64, u64x8, into_array_u64);
    test_swizzle2_0xcc!(u64, u64x8, into_array_u64);
    test_swizzle2_0xf0!(u64, u64x8, into_array_u64);
    network64bit1!(u64, u64x8, into_array_u64);
    network64bit2!(u64, u64x8, into_array_u64);
    network64bit3!(u64, u64x8, into_array_u64);
    network64bit4!(u64, u64x8, into_array_u64);
}
