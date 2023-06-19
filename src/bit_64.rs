use crate::{cmp_merge, coex, partition_avx512, SimdCompare, SimdSortable};

pub(crate) trait Bit64Element: SimdSortable {}

impl Bit64Element for u64 {}

impl Bit64Element for i64 {}

pub(crate) trait Bit64Simd<T: Bit64Element> {
    fn swizzle2_0xaa(a: Self, b: Self) -> Self;
    fn swizzle2_0xcc(a: Self, b: Self) -> Self;
    fn swizzle2_0xf0(a: Self, b: Self) -> Self;

    /// [1, 0, 3, 2, 5, 4, 7, 6]
    fn shuffle1_1_1_1(a: Self) -> Self;

    //   ZMM                    7, 6, 5, 4, 3, 2, 1, 0
    /// #define NETWORK_64BIT_1 4, 5, 6, 7, 0, 1, 2, 3
    fn network64bit1(a: Self) -> Self;
    //   ZMM                    7, 6, 5, 4, 3, 2, 1, 0
    /// #define NETWORK_64BIT_2 0, 1, 2, 3, 4, 5, 6, 7
    fn network64bit2(a: Self) -> Self;
    //   ZMM                    7, 6, 5, 4, 3, 2, 1, 0
    /// #define NETWORK_64BIT_3 5, 4, 7, 6, 1, 0, 3, 2
    fn network64bit3(a: Self) -> Self;
    //   ZMM                    7, 6, 5, 4, 3, 2, 1, 0
    /// #define NETWORK_64BIT_4 3, 2, 1, 0, 7, 6, 5, 4
    fn network64bit4(a: Self) -> Self;
}

/*
 * Assumes zmm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
#[inline]
pub(crate) fn sort_zmm_64bit<U, T>(mut zmm: T) -> T
where
    U: Bit64Element,
    T: SimdCompare<U, 8> + Bit64Simd<U>,
{
    zmm = cmp_merge(zmm, T::shuffle1_1_1_1(zmm), T::swizzle2_0xaa);
    zmm = cmp_merge(zmm, T::network64bit1(zmm), T::swizzle2_0xcc);
    zmm = cmp_merge(zmm, T::shuffle1_1_1_1(zmm), T::swizzle2_0xaa);
    zmm = cmp_merge(zmm, T::network64bit2(zmm), T::swizzle2_0xf0);
    zmm = cmp_merge(zmm, T::network64bit3(zmm), T::swizzle2_0xcc);
    zmm = cmp_merge(zmm, T::shuffle1_1_1_1(zmm), T::swizzle2_0xaa);
    zmm
}

// Assumes zmm is bitonic and performs a recursive half cleaner
#[inline]
fn bitonic_merge_zmm_64bit<U: Bit64Element, T: SimdCompare<U, 8> + Bit64Simd<U>>(mut zmm: T) -> T {
    // 1) half_cleaner[8]: compare 0-4, 1-5, 2-6, 3-7
    zmm = cmp_merge(zmm, T::network64bit4(zmm), T::swizzle2_0xf0);
    // 2) half_cleaner[4]
    zmm = cmp_merge(zmm, T::network64bit3(zmm), T::swizzle2_0xcc);
    // 3) half_cleaner[1]
    zmm = cmp_merge(zmm, T::shuffle1_1_1_1(zmm), T::swizzle2_0xaa);
    zmm
}

// Assumes zmm1 and zmm2 are sorted and performs a recursive half cleaner
#[inline]
fn bitonic_merge_two_zmm_64bit<U: Bit64Element, T: SimdCompare<U, 8> + Bit64Simd<U>>(
    zmm1: &mut T,
    zmm2: &mut T,
) {
    // 1) First step of a merging network: coex of zmm1 and zmm2 reversed
    *zmm2 = T::network64bit2(*zmm2);
    let zmm3 = T::min(*zmm1, *zmm2);
    let zmm4 = T::max(*zmm1, *zmm2);
    // 2) Recursive half cleaner for each
    *zmm1 = bitonic_merge_zmm_64bit(zmm3);
    *zmm2 = bitonic_merge_zmm_64bit(zmm4);
}

// Assumes [zmm0, zmm1] and [zmm2, zmm3] are sorted and performs a recursive
// half cleaner
#[inline]
fn bitonic_merge_four_zmm_64bit<U: Bit64Element, T: SimdCompare<U, 8> + Bit64Simd<U>>(
    zmm: &mut [T],
) {
    // 1) First step of a merging network
    let zmm2r = T::network64bit2(zmm[2]);
    let zmm3r = T::network64bit2(zmm[3]);
    let zmm_t1 = T::min(zmm[0], zmm3r);
    let zmm_t2 = T::min(zmm[1], zmm2r);
    // 2) Recursive half clearer: 16
    let zmm_t3 = T::network64bit2(T::max(zmm[1], zmm2r));
    let zmm_t4 = T::network64bit2(T::max(zmm[0], zmm3r));
    let zmm0 = T::min(zmm_t1, zmm_t2);
    let zmm1 = T::max(zmm_t1, zmm_t2);
    let zmm2 = T::min(zmm_t3, zmm_t4);
    let zmm3 = T::max(zmm_t3, zmm_t4);
    zmm[0] = bitonic_merge_zmm_64bit(zmm0);
    zmm[1] = bitonic_merge_zmm_64bit(zmm1);
    zmm[2] = bitonic_merge_zmm_64bit(zmm2);
    zmm[3] = bitonic_merge_zmm_64bit(zmm3);
}

#[inline]
fn bitonic_merge_eight_zmm_64bit<U: Bit64Element, T: SimdCompare<U, 8> + Bit64Simd<U>>(
    zmm: &mut [T],
) {
    let zmm4r = T::network64bit2(zmm[4]);
    let zmm5r = T::network64bit2(zmm[5]);
    let zmm6r = T::network64bit2(zmm[6]);
    let zmm7r = T::network64bit2(zmm[7]);
    let mut zmm_t1 = T::min(zmm[0], zmm7r);
    let mut zmm_t2 = T::min(zmm[1], zmm6r);
    let mut zmm_t3 = T::min(zmm[2], zmm5r);
    let mut zmm_t4 = T::min(zmm[3], zmm4r);
    let mut zmm_t5 = T::network64bit2(T::max(zmm[3], zmm4r));
    let mut zmm_t6 = T::network64bit2(T::max(zmm[2], zmm5r));
    let mut zmm_t7 = T::network64bit2(T::max(zmm[1], zmm6r));
    let mut zmm_t8 = T::network64bit2(T::max(zmm[0], zmm7r));
    coex(&mut zmm_t1, &mut zmm_t3);
    coex(&mut zmm_t2, &mut zmm_t4);
    coex(&mut zmm_t5, &mut zmm_t7);
    coex(&mut zmm_t6, &mut zmm_t8);
    coex(&mut zmm_t1, &mut zmm_t2);
    coex(&mut zmm_t3, &mut zmm_t4);
    coex(&mut zmm_t5, &mut zmm_t6);
    coex(&mut zmm_t7, &mut zmm_t8);
    zmm[0] = bitonic_merge_zmm_64bit(zmm_t1);
    zmm[1] = bitonic_merge_zmm_64bit(zmm_t2);
    zmm[2] = bitonic_merge_zmm_64bit(zmm_t3);
    zmm[3] = bitonic_merge_zmm_64bit(zmm_t4);
    zmm[4] = bitonic_merge_zmm_64bit(zmm_t5);
    zmm[5] = bitonic_merge_zmm_64bit(zmm_t6);
    zmm[6] = bitonic_merge_zmm_64bit(zmm_t7);
    zmm[7] = bitonic_merge_zmm_64bit(zmm_t8);
}

#[inline]
fn bitonic_merge_sixteen_zmm_64bit<U: Bit64Element, T: SimdCompare<U, 8> + Bit64Simd<U>>(
    zmm: &mut [T],
) {
    let zmm8r = T::network64bit2(zmm[8]);
    let zmm9r = T::network64bit2(zmm[9]);
    let zmm10r = T::network64bit2(zmm[10]);
    let zmm11r = T::network64bit2(zmm[11]);
    let zmm12r = T::network64bit2(zmm[12]);
    let zmm13r = T::network64bit2(zmm[13]);
    let zmm14r = T::network64bit2(zmm[14]);
    let zmm15r = T::network64bit2(zmm[15]);
    let mut zmm_t1 = T::min(zmm[0], zmm15r);
    let mut zmm_t2 = T::min(zmm[1], zmm14r);
    let mut zmm_t3 = T::min(zmm[2], zmm13r);
    let mut zmm_t4 = T::min(zmm[3], zmm12r);
    let mut zmm_t5 = T::min(zmm[4], zmm11r);
    let mut zmm_t6 = T::min(zmm[5], zmm10r);
    let mut zmm_t7 = T::min(zmm[6], zmm9r);
    let mut zmm_t8 = T::min(zmm[7], zmm8r);
    let mut zmm_t9 = T::network64bit2(T::max(zmm[7], zmm8r));
    let mut zmm_t10 = T::network64bit2(T::max(zmm[6], zmm9r));
    let mut zmm_t11 = T::network64bit2(T::max(zmm[5], zmm10r));
    let mut zmm_t12 = T::network64bit2(T::max(zmm[4], zmm11r));
    let mut zmm_t13 = T::network64bit2(T::max(zmm[3], zmm12r));
    let mut zmm_t14 = T::network64bit2(T::max(zmm[2], zmm13r));
    let mut zmm_t15 = T::network64bit2(T::max(zmm[1], zmm14r));
    let mut zmm_t16 = T::network64bit2(T::max(zmm[0], zmm15r));
    // Recusive half clear 16 zmm regs
    coex(&mut zmm_t1, &mut zmm_t5);
    coex(&mut zmm_t2, &mut zmm_t6);
    coex(&mut zmm_t3, &mut zmm_t7);
    coex(&mut zmm_t4, &mut zmm_t8);
    coex(&mut zmm_t9, &mut zmm_t13);
    coex(&mut zmm_t10, &mut zmm_t14);
    coex(&mut zmm_t11, &mut zmm_t15);
    coex(&mut zmm_t12, &mut zmm_t16);
    //
    coex(&mut zmm_t1, &mut zmm_t3);
    coex(&mut zmm_t2, &mut zmm_t4);
    coex(&mut zmm_t5, &mut zmm_t7);
    coex(&mut zmm_t6, &mut zmm_t8);
    coex(&mut zmm_t9, &mut zmm_t11);
    coex(&mut zmm_t10, &mut zmm_t12);
    coex(&mut zmm_t13, &mut zmm_t15);
    coex(&mut zmm_t14, &mut zmm_t16);
    //
    coex(&mut zmm_t1, &mut zmm_t2);
    coex(&mut zmm_t3, &mut zmm_t4);
    coex(&mut zmm_t5, &mut zmm_t6);
    coex(&mut zmm_t7, &mut zmm_t8);
    coex(&mut zmm_t9, &mut zmm_t10);
    coex(&mut zmm_t11, &mut zmm_t12);
    coex(&mut zmm_t13, &mut zmm_t14);
    coex(&mut zmm_t15, &mut zmm_t16);
    //
    zmm[0] = bitonic_merge_zmm_64bit(zmm_t1);
    zmm[1] = bitonic_merge_zmm_64bit(zmm_t2);
    zmm[2] = bitonic_merge_zmm_64bit(zmm_t3);
    zmm[3] = bitonic_merge_zmm_64bit(zmm_t4);
    zmm[4] = bitonic_merge_zmm_64bit(zmm_t5);
    zmm[5] = bitonic_merge_zmm_64bit(zmm_t6);
    zmm[6] = bitonic_merge_zmm_64bit(zmm_t7);
    zmm[7] = bitonic_merge_zmm_64bit(zmm_t8);
    zmm[8] = bitonic_merge_zmm_64bit(zmm_t9);
    zmm[9] = bitonic_merge_zmm_64bit(zmm_t10);
    zmm[10] = bitonic_merge_zmm_64bit(zmm_t11);
    zmm[11] = bitonic_merge_zmm_64bit(zmm_t12);
    zmm[12] = bitonic_merge_zmm_64bit(zmm_t13);
    zmm[13] = bitonic_merge_zmm_64bit(zmm_t14);
    zmm[14] = bitonic_merge_zmm_64bit(zmm_t15);
    zmm[15] = bitonic_merge_zmm_64bit(zmm_t16);
}

pub(crate) fn sort_8<T, U>(data: &mut [T])
where
    T: Bit64Element,
    U: Bit64Simd<T> + SimdCompare<T, 8>,
{
    let simd = U::loadu(data);
    let zmm = sort_zmm_64bit(simd);
    U::storeu(zmm, data);
}

pub(crate) fn sort_16<T, U>(data: &mut [T])
where
    T: Bit64Element,
    U: Bit64Simd<T> + SimdCompare<T, 8>,
{
    if data.len() <= 8 {
        sort_8::<T, U>(data);
    } else {
        let zmm = data.split_at_mut(8);
        let mut zmm1 = U::loadu(zmm.0);
        let mut zmm2 = U::loadu(zmm.1);
        zmm1 = sort_zmm_64bit(zmm1);
        zmm2 = sort_zmm_64bit(zmm2);

        bitonic_merge_two_zmm_64bit(&mut zmm1, &mut zmm2);
        U::storeu(zmm1, zmm.0);
        U::storeu(zmm2, zmm.1);
    }
}

pub(crate) fn sort_32<T, U>(data: &mut [T])
where
    T: Bit64Element,
    U: Bit64Simd<T> + SimdCompare<T, 8>,
{
    let mut max_value_array: [T; 8] = [T::MAX_VALUE; 8];
    if data.len() <= 16 {
        sort_16::<T, U>(data);
        return;
    };
    let (data_16_0, data_16_1) = data.split_at_mut(16);
    let (data_8_0, data_8_1) = data_16_0.split_at_mut(8);
    let (data_8_2, data_8_3) = {
        if data_16_1.len() > 8 {
            data_16_1.split_at_mut(8)
        } else {
            (data_16_1, max_value_array.as_mut_slice())
        }
    };

    let mut zmm_0 = U::loadu(data_8_0);
    let mut zmm_1 = U::loadu(data_8_1);
    let mut zmm_2 = U::loadu(data_8_2);
    let mut zmm_3 = U::loadu(data_8_3);

    zmm_0 = sort_zmm_64bit(zmm_0);
    zmm_1 = sort_zmm_64bit(zmm_1);
    zmm_2 = sort_zmm_64bit(zmm_2);
    zmm_3 = sort_zmm_64bit(zmm_3);
    bitonic_merge_two_zmm_64bit(&mut zmm_0, &mut zmm_1);
    bitonic_merge_two_zmm_64bit(&mut zmm_2, &mut zmm_3);

    let mut zmm = [zmm_0, zmm_1, zmm_2, zmm_3];
    bitonic_merge_four_zmm_64bit(&mut zmm);

    U::storeu(zmm[0], data_8_0);
    U::storeu(zmm[1], data_8_1);
    U::storeu(zmm[2], data_8_2);
    U::storeu(zmm[3], data_8_3);
}

pub(crate) fn sort_64<T, U>(data: &mut [T])
where
    T: Bit64Element,
    U: Bit64Simd<T> + SimdCompare<T, 8>,
{
    if data.len() <= 32 {
        sort_32::<T, U>(data);
        return;
    }
    let (data_32_0, data_32_1) = data.split_at_mut(32);
    let (data_16_0, data_16_1) = data_32_0.split_at_mut(16);
    let (data_8_0, data_8_1) = data_16_0.split_at_mut(8);
    let (data_8_2, data_8_3) = data_16_1.split_at_mut(8);
    let mut zmm_0 = U::loadu(data_8_0);
    let mut zmm_1 = U::loadu(data_8_1);
    let mut zmm_2 = U::loadu(data_8_2);
    let mut zmm_3 = U::loadu(data_8_3);
    zmm_0 = sort_zmm_64bit(zmm_0);
    zmm_1 = sort_zmm_64bit(zmm_1);
    zmm_2 = sort_zmm_64bit(zmm_2);
    zmm_3 = sort_zmm_64bit(zmm_3);

    let mut max_value_array: [T; 24] = [T::MAX_VALUE; 24];
    let max_value_array_t = &mut max_value_array[..];
    let (data_8_4, data_8_5, data_8_6, data_8_7) = {
        if data_32_1.len() < 8 {
            let (x, max_value_array_t) = max_value_array_t.split_at_mut(8);
            let (y, max_value_array_t) = max_value_array_t.split_at_mut(8);
            let (z, _) = max_value_array_t.split_at_mut(8);
            (data_32_1, x, y, z)
        } else if data_32_1.len() < 16 {
            let (x, y) = data_32_1.split_at_mut(8);
            let (z, max_value_array_t) = max_value_array_t.split_at_mut(8);
            let (w, _) = max_value_array_t.split_at_mut(8);
            (x, y, z, w)
        } else if data_32_1.len() < 24 {
            let (x, data_32_1) = data_32_1.split_at_mut(8);
            let (y, z) = data_32_1.split_at_mut(8);
            let (w, _) = max_value_array_t.split_at_mut(8);
            (x, y, z, w)
        } else {
            let (x, data_32_1) = data_32_1.split_at_mut(8);
            let (y, data_32_1) = data_32_1.split_at_mut(8);
            let (z, w) = data_32_1.split_at_mut(8);
            (x, y, z, w)
        }
    };
    let mut zmm_4 = U::loadu(data_8_4);
    let mut zmm_5 = U::loadu(data_8_5);
    let mut zmm_6 = U::loadu(data_8_6);
    let mut zmm_7 = U::loadu(data_8_7);
    zmm_4 = sort_zmm_64bit(zmm_4);
    zmm_5 = sort_zmm_64bit(zmm_5);
    zmm_6 = sort_zmm_64bit(zmm_6);
    zmm_7 = sort_zmm_64bit(zmm_7);
    bitonic_merge_two_zmm_64bit(&mut zmm_0, &mut zmm_1);
    bitonic_merge_two_zmm_64bit(&mut zmm_2, &mut zmm_3);
    bitonic_merge_two_zmm_64bit(&mut zmm_4, &mut zmm_5);
    bitonic_merge_two_zmm_64bit(&mut zmm_6, &mut zmm_7);
    let mut zmm = [zmm_0, zmm_1, zmm_2, zmm_3];
    bitonic_merge_four_zmm_64bit(&mut zmm);

    let mut zmm_1 = [zmm_4, zmm_5, zmm_6, zmm_7];
    bitonic_merge_four_zmm_64bit(&mut zmm_1);

    let mut zmm = [
        zmm[0], zmm[1], zmm[2], zmm[3], zmm_1[0], zmm_1[1], zmm_1[2], zmm_1[3],
    ];
    bitonic_merge_eight_zmm_64bit(&mut zmm);
    U::storeu(zmm[0], data_8_0);
    U::storeu(zmm[1], data_8_1);
    U::storeu(zmm[2], data_8_2);
    U::storeu(zmm[3], data_8_3);
    U::storeu(zmm[4], data_8_4);
    U::storeu(zmm[5], data_8_5);
    U::storeu(zmm[6], data_8_6);
    U::storeu(zmm[7], data_8_7);
}

pub(crate) fn sort_128<T, U>(data: &mut [T])
where
    T: Bit64Element,
    U: Bit64Simd<T> + SimdCompare<T, 8>,
{
    if data.len() <= 64 {
        sort_64::<T, U>(data);
        return;
    }

    let (data_64_0, data_64_1) = data.split_at_mut(64);
    let (data_8_0, data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7) =
        split_64::<T, U>(data_64_0, &mut []);
    let mut zmm_0 = U::loadu(data_8_0);
    let mut zmm_1 = U::loadu(data_8_1);
    let mut zmm_2 = U::loadu(data_8_2);
    let mut zmm_3 = U::loadu(data_8_3);
    let mut zmm_4 = U::loadu(data_8_4);
    let mut zmm_5 = U::loadu(data_8_5);
    let mut zmm_6 = U::loadu(data_8_6);
    let mut zmm_7 = U::loadu(data_8_7);
    zmm_0 = sort_zmm_64bit(zmm_0);
    zmm_1 = sort_zmm_64bit(zmm_1);
    zmm_2 = sort_zmm_64bit(zmm_2);
    zmm_3 = sort_zmm_64bit(zmm_3);
    zmm_4 = sort_zmm_64bit(zmm_4);
    zmm_5 = sort_zmm_64bit(zmm_5);
    zmm_6 = sort_zmm_64bit(zmm_6);
    zmm_7 = sort_zmm_64bit(zmm_7);

    let mut max_value_array: [T; 64] = [T::MAX_VALUE; 64];
    let (data_8_8, data_8_9, data_8_10, data_8_11, data_8_12, data_8_13, data_8_14, data_8_15) =
        split_64::<T, U>(data_64_1, &mut max_value_array);
    let mut zmm_8 = U::loadu(data_8_8);
    let mut zmm_9 = U::loadu(data_8_9);
    let mut zmm_10 = U::loadu(data_8_10);
    let mut zmm_11 = U::loadu(data_8_11);
    let mut zmm_12 = U::loadu(data_8_12);
    let mut zmm_13 = U::loadu(data_8_13);
    let mut zmm_14 = U::loadu(data_8_14);
    let mut zmm_15 = U::loadu(data_8_15);
    zmm_8 = sort_zmm_64bit(zmm_8);
    zmm_9 = sort_zmm_64bit(zmm_9);
    zmm_10 = sort_zmm_64bit(zmm_10);
    zmm_11 = sort_zmm_64bit(zmm_11);
    zmm_12 = sort_zmm_64bit(zmm_12);
    zmm_13 = sort_zmm_64bit(zmm_13);
    zmm_14 = sort_zmm_64bit(zmm_14);
    zmm_15 = sort_zmm_64bit(zmm_15);
    bitonic_merge_two_zmm_64bit(&mut zmm_0, &mut zmm_1);
    bitonic_merge_two_zmm_64bit(&mut zmm_2, &mut zmm_3);
    bitonic_merge_two_zmm_64bit(&mut zmm_4, &mut zmm_5);
    bitonic_merge_two_zmm_64bit(&mut zmm_6, &mut zmm_7);
    bitonic_merge_two_zmm_64bit(&mut zmm_8, &mut zmm_9);
    bitonic_merge_two_zmm_64bit(&mut zmm_10, &mut zmm_11);
    bitonic_merge_two_zmm_64bit(&mut zmm_12, &mut zmm_13);
    bitonic_merge_two_zmm_64bit(&mut zmm_14, &mut zmm_15);
    let mut zmm_4_1 = [zmm_0, zmm_1, zmm_2, zmm_3];
    bitonic_merge_four_zmm_64bit(&mut zmm_4_1);
    let mut zmm_4_2 = [zmm_4, zmm_5, zmm_6, zmm_7];
    bitonic_merge_four_zmm_64bit(&mut zmm_4_2);
    let mut zmm_4_3 = [zmm_8, zmm_9, zmm_10, zmm_11];
    bitonic_merge_four_zmm_64bit(&mut zmm_4_3);
    let mut zmm_4_4 = [zmm_12, zmm_13, zmm_14, zmm_15];
    bitonic_merge_four_zmm_64bit(&mut zmm_4_4);
    let mut zmm_8_1 = [
        zmm_4_1[0], zmm_4_1[1], zmm_4_1[2], zmm_4_1[3], zmm_4_2[0], zmm_4_2[1], zmm_4_2[2],
        zmm_4_2[3],
    ];
    bitonic_merge_eight_zmm_64bit(&mut zmm_8_1);
    let mut zmm_8_2 = [
        zmm_4_3[0], zmm_4_3[1], zmm_4_3[2], zmm_4_3[3], zmm_4_4[0], zmm_4_4[1], zmm_4_4[2],
        zmm_4_4[3],
    ];
    bitonic_merge_eight_zmm_64bit(&mut zmm_8_2);
    let mut zmm = [
        zmm_8_1[0], zmm_8_1[1], zmm_8_1[2], zmm_8_1[3], zmm_8_1[4], zmm_8_1[5], zmm_8_1[6],
        zmm_8_1[7], zmm_8_2[0], zmm_8_2[1], zmm_8_2[2], zmm_8_2[3], zmm_8_2[4], zmm_8_2[5],
        zmm_8_2[6], zmm_8_2[7],
    ];
    bitonic_merge_sixteen_zmm_64bit(&mut zmm);

    U::storeu(zmm[0], data_8_0);
    U::storeu(zmm[1], data_8_1);
    U::storeu(zmm[2], data_8_2);
    U::storeu(zmm[3], data_8_3);
    U::storeu(zmm[4], data_8_4);
    U::storeu(zmm[5], data_8_5);
    U::storeu(zmm[6], data_8_6);
    U::storeu(zmm[7], data_8_7);
    U::storeu(zmm[8], data_8_8);
    U::storeu(zmm[9], data_8_9);
    U::storeu(zmm[10], data_8_10);
    U::storeu(zmm[11], data_8_11);
    U::storeu(zmm[12], data_8_12);
    U::storeu(zmm[13], data_8_13);
    U::storeu(zmm[14], data_8_14);
    U::storeu(zmm[15], data_8_15);
}

#[inline]
fn split_64<'a, T, U>(
    data: &'a mut [T],
    max_value_array_t: &'a mut [T],
) -> (
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
)
where
    T: Bit64Element,
    U: SimdCompare<T, 8> + Bit64Simd<T>,
{
    if data.len() > 8 * 7 {
        let (data_8_1, data) = data.split_at_mut(8);
        let (data_8_2, data) = data.split_at_mut(8);
        let (data_8_3, data) = data.split_at_mut(8);
        let (data_8_4, data) = data.split_at_mut(8);
        let (data_8_5, data) = data.split_at_mut(8);
        let (data_8_6, data) = data.split_at_mut(8);
        let (data_8_7, data_8_8) = data.split_at_mut(8);
        (
            data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7, data_8_8,
        )
    } else if data.len() > 8 * 6 {
        let (data_8_1, data) = data.split_at_mut(8);
        let (data_8_2, data) = data.split_at_mut(8);
        let (data_8_3, data) = data.split_at_mut(8);
        let (data_8_4, data) = data.split_at_mut(8);
        let (data_8_5, data) = data.split_at_mut(8);
        let (data_8_6, data_8_7) = data.split_at_mut(8);
        let (data_8_8, _) = max_value_array_t.split_at_mut(8);
        (
            data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7, data_8_8,
        )
    } else if data.len() > 8 * 5 {
        let (data_8_1, data) = data.split_at_mut(8);
        let (data_8_2, data) = data.split_at_mut(8);
        let (data_8_3, data) = data.split_at_mut(8);
        let (data_8_4, data) = data.split_at_mut(8);
        let (data_8_5, data_8_6) = data.split_at_mut(8);
        let (data_8_7, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_8, _) = max_value_array_t.split_at_mut(8);
        (
            data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7, data_8_8,
        )
    } else if data.len() > 8 * 4 {
        let (data_8_1, data) = data.split_at_mut(8);
        let (data_8_2, data) = data.split_at_mut(8);
        let (data_8_3, data) = data.split_at_mut(8);
        let (data_8_4, data_8_5) = data.split_at_mut(8);
        let (data_8_6, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_7, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_8, _) = max_value_array_t.split_at_mut(8);
        (
            data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7, data_8_8,
        )
    } else if data.len() > 8 * 3 {
        let (data_8_1, data) = data.split_at_mut(8);
        let (data_8_2, data) = data.split_at_mut(8);
        let (data_8_3, data_8_4) = data.split_at_mut(8);
        let (data_8_5, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_6, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_7, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_8, _) = max_value_array_t.split_at_mut(8);
        (
            data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7, data_8_8,
        )
    } else if data.len() > 8 * 2 {
        let (data_8_1, data) = data.split_at_mut(8);
        let (data_8_2, data_8_3) = data.split_at_mut(8);
        let (data_8_4, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_5, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_6, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_7, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_8, _) = max_value_array_t.split_at_mut(8);
        (
            data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7, data_8_8,
        )
    } else if data.len() > 8 * 1 {
        let (data_8_1, data_8_2) = data.split_at_mut(8);
        let (data_8_3, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_4, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_5, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_6, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_7, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_8, _) = max_value_array_t.split_at_mut(8);
        (
            data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7, data_8_8,
        )
    } else {
        let data_8_1 = data;
        let (data_8_2, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_3, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_4, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_5, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_6, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_7, max_value_array_t) = max_value_array_t.split_at_mut(8);
        let (data_8_8, _) = max_value_array_t.split_at_mut(8);
        (
            data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7, data_8_8,
        )
    }
}

fn get_pivot_64bit<T, U>(data: &[T]) -> T
where
    T: Bit64Element,
    U: SimdCompare<T, 8> + Bit64Simd<T>,
{
    // median of 8
    let size = data.len() / 8;
    let rand_index = [
        size,
        2 * size,
        3 * size,
        4 * size,
        5 * size,
        6 * size,
        7 * size,
        8 * size,
    ];

    let rand_vec = U::gather_from_idx(rand_index, data);
    // pivot will never be a nan, since there are no nan's!
    let sort = sort_zmm_64bit(rand_vec);
    return U::get_value_at_idx(sort, 4);
}

pub(crate) fn qsort_64bit_<T, U>(data: &mut [T], max_iters: i64)
where
    T: Bit64Element,
    U: SimdCompare<T, 8> + Bit64Simd<T>,
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if max_iters <= 0 {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= 128
     */
    if data.len() <= 128 {
        sort_128::<T, U>(data);
        return;
    }

    let pivot = get_pivot_64bit::<T, U>(data);
    let mut smallest = T::MAX_VALUE;
    let mut biggest = T::MIN_VALUE;
    let pivot_index = partition_avx512::<T, 8, U>(data, pivot, &mut smallest, &mut biggest);
    let (left, right) = data.split_at_mut(pivot_index);
    if pivot != smallest {
        qsort_64bit_::<T, U>(left, max_iters - 1);
    }
    if pivot != biggest {
        qsort_64bit_::<T, U>(right, max_iters - 1);
    }
}
