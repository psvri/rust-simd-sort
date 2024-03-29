use std::cmp;

use crate::{cmp_merge, coex, partition_avx512_unrolled, SimdCompare, SimdSortable};

pub trait Bit64Element: SimdSortable {}

impl Bit64Element for u64 {}

impl Bit64Element for i64 {}
impl Bit64Element for f64 {}

pub(crate) trait Bit64Simd<T: Bit64Element> {
    //ZMM 76543210
    //    10101010 , 0 -> first, 1 -> second
    fn swizzle2_0xaa(a: Self, b: Self) -> Self;

    //ZMM 76543210
    //    11001100, 0 -> first, 1 -> second
    fn swizzle2_0xcc(a: Self, b: Self) -> Self;

    //ZMM 76543210
    //    11110000, 0 -> first, 1 -> second
    fn swizzle2_0xf0(a: Self, b: Self) -> Self;

    /// [1, 0, 3, 2, 5, 4, 7, 6]
    fn shuffle1_1_1_1(a: Self) -> Self;

    ///   ZMM                    7, 6, 5, 4, 3, 2, 1, 0
    ///  #define NETWORK_64BIT_1 4, 5, 6, 7, 0, 1, 2, 3
    fn network64bit1(a: Self) -> Self;
    ///   ZMM                    7, 6, 5, 4, 3, 2, 1, 0
    ///  #define NETWORK_64BIT_2 0, 1, 2, 3, 4, 5, 6, 7
    fn network64bit2(a: Self) -> Self;
    ///   ZMM                    7, 6, 5, 4, 3, 2, 1, 0
    ///  #define NETWORK_64BIT_3 5, 4, 7, 6, 1, 0, 3, 2
    fn network64bit3(a: Self) -> Self;
    ///   ZMM                    7, 6, 5, 4, 3, 2, 1, 0
    ///  #define NETWORK_64BIT_4 3, 2, 1, 0, 7, 6, 5, 4
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

#[inline]
fn bitonic_merge_thirtytwo_zmm_64bit<U: Bit64Element, T: SimdCompare<U, 8> + Bit64Simd<U>>(
    zmm: &mut [T],
) {
    let zmm16r = T::network64bit2(zmm[16]);
    let zmm17r = T::network64bit2(zmm[17]);
    let zmm18r = T::network64bit2(zmm[18]);
    let zmm19r = T::network64bit2(zmm[19]);
    let zmm20r = T::network64bit2(zmm[20]);
    let zmm21r = T::network64bit2(zmm[21]);
    let zmm22r = T::network64bit2(zmm[22]);
    let zmm23r = T::network64bit2(zmm[23]);
    let zmm24r = T::network64bit2(zmm[24]);
    let zmm25r = T::network64bit2(zmm[25]);
    let zmm26r = T::network64bit2(zmm[26]);
    let zmm27r = T::network64bit2(zmm[27]);
    let zmm28r = T::network64bit2(zmm[28]);
    let zmm29r = T::network64bit2(zmm[29]);
    let zmm30r = T::network64bit2(zmm[30]);
    let zmm31r = T::network64bit2(zmm[31]);
    let mut zmm_t1 = T::min(zmm[0], zmm31r);
    let mut zmm_t2 = T::min(zmm[1], zmm30r);
    let mut zmm_t3 = T::min(zmm[2], zmm29r);
    let mut zmm_t4 = T::min(zmm[3], zmm28r);
    let mut zmm_t5 = T::min(zmm[4], zmm27r);
    let mut zmm_t6 = T::min(zmm[5], zmm26r);
    let mut zmm_t7 = T::min(zmm[6], zmm25r);
    let mut zmm_t8 = T::min(zmm[7], zmm24r);
    let mut zmm_t9 = T::min(zmm[8], zmm23r);
    let mut zmm_t10 = T::min(zmm[9], zmm22r);
    let mut zmm_t11 = T::min(zmm[10], zmm21r);
    let mut zmm_t12 = T::min(zmm[11], zmm20r);
    let mut zmm_t13 = T::min(zmm[12], zmm19r);
    let mut zmm_t14 = T::min(zmm[13], zmm18r);
    let mut zmm_t15 = T::min(zmm[14], zmm17r);
    let mut zmm_t16 = T::min(zmm[15], zmm16r);
    let mut zmm_t17 = T::network64bit2(T::max(zmm[15], zmm16r));
    let mut zmm_t18 = T::network64bit2(T::max(zmm[14], zmm17r));
    let mut zmm_t19 = T::network64bit2(T::max(zmm[13], zmm18r));
    let mut zmm_t20 = T::network64bit2(T::max(zmm[12], zmm19r));
    let mut zmm_t21 = T::network64bit2(T::max(zmm[11], zmm20r));
    let mut zmm_t22 = T::network64bit2(T::max(zmm[10], zmm21r));
    let mut zmm_t23 = T::network64bit2(T::max(zmm[9], zmm22r));
    let mut zmm_t24 = T::network64bit2(T::max(zmm[8], zmm23r));
    let mut zmm_t25 = T::network64bit2(T::max(zmm[7], zmm24r));
    let mut zmm_t26 = T::network64bit2(T::max(zmm[6], zmm25r));
    let mut zmm_t27 = T::network64bit2(T::max(zmm[5], zmm26r));
    let mut zmm_t28 = T::network64bit2(T::max(zmm[4], zmm27r));
    let mut zmm_t29 = T::network64bit2(T::max(zmm[3], zmm28r));
    let mut zmm_t30 = T::network64bit2(T::max(zmm[2], zmm29r));
    let mut zmm_t31 = T::network64bit2(T::max(zmm[1], zmm30r));
    let mut zmm_t32 = T::network64bit2(T::max(zmm[0], zmm31r));
    // Recusive half clear 16 zmm regs
    coex(&mut zmm_t1, &mut zmm_t9);
    coex(&mut zmm_t2, &mut zmm_t10);
    coex(&mut zmm_t3, &mut zmm_t11);
    coex(&mut zmm_t4, &mut zmm_t12);
    coex(&mut zmm_t5, &mut zmm_t13);
    coex(&mut zmm_t6, &mut zmm_t14);
    coex(&mut zmm_t7, &mut zmm_t15);
    coex(&mut zmm_t8, &mut zmm_t16);
    coex(&mut zmm_t17, &mut zmm_t25);
    coex(&mut zmm_t18, &mut zmm_t26);
    coex(&mut zmm_t19, &mut zmm_t27);
    coex(&mut zmm_t20, &mut zmm_t28);
    coex(&mut zmm_t21, &mut zmm_t29);
    coex(&mut zmm_t22, &mut zmm_t30);
    coex(&mut zmm_t23, &mut zmm_t31);
    coex(&mut zmm_t24, &mut zmm_t32);
    //
    coex(&mut zmm_t1, &mut zmm_t5);
    coex(&mut zmm_t2, &mut zmm_t6);
    coex(&mut zmm_t3, &mut zmm_t7);
    coex(&mut zmm_t4, &mut zmm_t8);
    coex(&mut zmm_t9, &mut zmm_t13);
    coex(&mut zmm_t10, &mut zmm_t14);
    coex(&mut zmm_t11, &mut zmm_t15);
    coex(&mut zmm_t12, &mut zmm_t16);
    coex(&mut zmm_t17, &mut zmm_t21);
    coex(&mut zmm_t18, &mut zmm_t22);
    coex(&mut zmm_t19, &mut zmm_t23);
    coex(&mut zmm_t20, &mut zmm_t24);
    coex(&mut zmm_t25, &mut zmm_t29);
    coex(&mut zmm_t26, &mut zmm_t30);
    coex(&mut zmm_t27, &mut zmm_t31);
    coex(&mut zmm_t28, &mut zmm_t32);
    //
    coex(&mut zmm_t1, &mut zmm_t3);
    coex(&mut zmm_t2, &mut zmm_t4);
    coex(&mut zmm_t5, &mut zmm_t7);
    coex(&mut zmm_t6, &mut zmm_t8);
    coex(&mut zmm_t9, &mut zmm_t11);
    coex(&mut zmm_t10, &mut zmm_t12);
    coex(&mut zmm_t13, &mut zmm_t15);
    coex(&mut zmm_t14, &mut zmm_t16);
    coex(&mut zmm_t17, &mut zmm_t19);
    coex(&mut zmm_t18, &mut zmm_t20);
    coex(&mut zmm_t21, &mut zmm_t23);
    coex(&mut zmm_t22, &mut zmm_t24);
    coex(&mut zmm_t25, &mut zmm_t27);
    coex(&mut zmm_t26, &mut zmm_t28);
    coex(&mut zmm_t29, &mut zmm_t31);
    coex(&mut zmm_t30, &mut zmm_t32);
    //
    coex(&mut zmm_t1, &mut zmm_t2);
    coex(&mut zmm_t3, &mut zmm_t4);
    coex(&mut zmm_t5, &mut zmm_t6);
    coex(&mut zmm_t7, &mut zmm_t8);
    coex(&mut zmm_t9, &mut zmm_t10);
    coex(&mut zmm_t11, &mut zmm_t12);
    coex(&mut zmm_t13, &mut zmm_t14);
    coex(&mut zmm_t15, &mut zmm_t16);
    coex(&mut zmm_t17, &mut zmm_t18);
    coex(&mut zmm_t19, &mut zmm_t20);
    coex(&mut zmm_t21, &mut zmm_t22);
    coex(&mut zmm_t23, &mut zmm_t24);
    coex(&mut zmm_t25, &mut zmm_t26);
    coex(&mut zmm_t27, &mut zmm_t28);
    coex(&mut zmm_t29, &mut zmm_t30);
    coex(&mut zmm_t31, &mut zmm_t32);
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
    zmm[16] = bitonic_merge_zmm_64bit(zmm_t17);
    zmm[17] = bitonic_merge_zmm_64bit(zmm_t18);
    zmm[18] = bitonic_merge_zmm_64bit(zmm_t19);
    zmm[19] = bitonic_merge_zmm_64bit(zmm_t20);
    zmm[20] = bitonic_merge_zmm_64bit(zmm_t21);
    zmm[21] = bitonic_merge_zmm_64bit(zmm_t22);
    zmm[22] = bitonic_merge_zmm_64bit(zmm_t23);
    zmm[23] = bitonic_merge_zmm_64bit(zmm_t24);
    zmm[24] = bitonic_merge_zmm_64bit(zmm_t25);
    zmm[25] = bitonic_merge_zmm_64bit(zmm_t26);
    zmm[26] = bitonic_merge_zmm_64bit(zmm_t27);
    zmm[27] = bitonic_merge_zmm_64bit(zmm_t28);
    zmm[28] = bitonic_merge_zmm_64bit(zmm_t29);
    zmm[29] = bitonic_merge_zmm_64bit(zmm_t30);
    zmm[30] = bitonic_merge_zmm_64bit(zmm_t31);
    zmm[31] = bitonic_merge_zmm_64bit(zmm_t32);
}

pub(crate) fn sort_8<T, U>(data: &mut [T])
where
    T: Bit64Element,
    U: Bit64Simd<T> + SimdCompare<T, 8>,
{
    let simd = U::mask_loadu(data);
    let zmm = sort_zmm_64bit(simd);
    U::mask_storeu(zmm, data);
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
        let mut zmm2 = U::mask_loadu(zmm.1);
        zmm1 = sort_zmm_64bit(zmm1);
        zmm2 = sort_zmm_64bit(zmm2);

        bitonic_merge_two_zmm_64bit(&mut zmm1, &mut zmm2);
        U::storeu(zmm1, zmm.0);
        U::mask_storeu(zmm2, zmm.1);
    }
}

pub(crate) fn sort_32<T, U>(data: &mut [T])
where
    T: Bit64Element,
    U: Bit64Simd<T> + SimdCompare<T, 8>,
{
    if data.len() <= 16 {
        sort_16::<T, U>(data);
        return;
    };
    let (data_16_0, data_16_1) = data.split_at_mut(16);
    let (data_8_0, data_8_1) = data_16_0.split_at_mut(8);
    let index_2 = cmp::min(data_16_1.len(), 8);
    let (data_8_2, data_8_3) = data_16_1.split_at_mut(index_2);

    let mut zmm_0 = U::loadu(data_8_0);
    let mut zmm_1 = U::loadu(data_8_1);
    let mut zmm_2 = U::mask_loadu(data_8_2);
    let mut zmm_3 = U::mask_loadu(data_8_3);

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
    U::mask_storeu(zmm[2], data_8_2);
    U::mask_storeu(zmm[3], data_8_3);
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

    let split_index = cmp::min(data_32_1.len(), 8);
    let (data_8_4, data_32_1) = data_32_1.split_at_mut(split_index);

    let split_index = cmp::min(data_32_1.len(), 8);
    let (data_8_5, data_32_1) = data_32_1.split_at_mut(split_index);

    let split_index = cmp::min(data_32_1.len(), 8);
    let (data_8_6, data_8_7) = data_32_1.split_at_mut(split_index);

    let mut zmm_4 = U::mask_loadu(data_8_4);
    let mut zmm_5 = U::mask_loadu(data_8_5);
    let mut zmm_6 = U::mask_loadu(data_8_6);
    let mut zmm_7 = U::mask_loadu(data_8_7);
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
    U::mask_storeu(zmm[4], data_8_4);
    U::mask_storeu(zmm[5], data_8_5);
    U::mask_storeu(zmm[6], data_8_6);
    U::mask_storeu(zmm[7], data_8_7);
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
    let split_index = cmp::min(data_64_0.len(), 8);
    let (data_8_0, data_64_0) = data_64_0.split_at_mut(split_index);

    let split_index = cmp::min(data_64_0.len(), 8);
    let (data_8_1, data_64_0) = data_64_0.split_at_mut(split_index);

    let split_index = cmp::min(data_64_0.len(), 8);
    let (data_8_2, data_64_0) = data_64_0.split_at_mut(split_index);

    let split_index = cmp::min(data_64_0.len(), 8);
    let (data_8_3, data_64_0) = data_64_0.split_at_mut(split_index);

    let split_index = cmp::min(data_64_0.len(), 8);
    let (data_8_4, data_64_0) = data_64_0.split_at_mut(split_index);

    let split_index = cmp::min(data_64_0.len(), 8);
    let (data_8_5, data_64_0) = data_64_0.split_at_mut(split_index);

    let split_index = cmp::min(data_64_0.len(), 8);
    let (data_8_6, data_8_7) = data_64_0.split_at_mut(split_index);

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

    let split_index = cmp::min(data_64_1.len(), 8);
    let (data_8_8, data_64_1) = data_64_1.split_at_mut(split_index);

    let split_index = cmp::min(data_64_1.len(), 8);
    let (data_8_9, data_64_1) = data_64_1.split_at_mut(split_index);

    let split_index = cmp::min(data_64_1.len(), 8);
    let (data_8_10, data_64_1) = data_64_1.split_at_mut(split_index);

    let split_index = cmp::min(data_64_1.len(), 8);
    let (data_8_11, data_64_1) = data_64_1.split_at_mut(split_index);

    let split_index = cmp::min(data_64_1.len(), 8);
    let (data_8_12, data_64_1) = data_64_1.split_at_mut(split_index);

    let split_index = cmp::min(data_64_1.len(), 8);
    let (data_8_13, data_64_1) = data_64_1.split_at_mut(split_index);

    let split_index = cmp::min(data_64_1.len(), 8);
    let (data_8_14, data_8_15) = data_64_1.split_at_mut(split_index);

    let mut zmm_8 = U::mask_loadu(data_8_8);
    let mut zmm_9 = U::mask_loadu(data_8_9);
    let mut zmm_10 = U::mask_loadu(data_8_10);
    let mut zmm_11 = U::mask_loadu(data_8_11);
    let mut zmm_12 = U::mask_loadu(data_8_12);
    let mut zmm_13 = U::mask_loadu(data_8_13);
    let mut zmm_14 = U::mask_loadu(data_8_14);
    let mut zmm_15 = U::mask_loadu(data_8_15);
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
    U::mask_storeu(zmm[8], data_8_8);
    U::mask_storeu(zmm[9], data_8_9);
    U::mask_storeu(zmm[10], data_8_10);
    U::mask_storeu(zmm[11], data_8_11);
    U::mask_storeu(zmm[12], data_8_12);
    U::mask_storeu(zmm[13], data_8_13);
    U::mask_storeu(zmm[14], data_8_14);
    U::mask_storeu(zmm[15], data_8_15);
}

pub(crate) fn sort_256<T, U>(data: &mut [T])
where
    T: Bit64Element,
    U: Bit64Simd<T> + SimdCompare<T, 8>,
{
    let n = data.len();
    if n <= 128 {
        sort_128::<T, U>(data);
        return;
    }

    let (data_128_0, data_128_1) = data.split_at_mut(128);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_0, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_1, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_2, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_3, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_4, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_5, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_6, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_7, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_8, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_9, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_10, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_11, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_12, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_13, data_128_0) = data_128_0.split_at_mut(split_index);
    let split_index = cmp::min(data_128_0.len(), 8);
    let (data_8_14, data_8_15) = data_128_0.split_at_mut(split_index);

    let mut zmm_0 = U::loadu(data_8_0);
    let mut zmm_1 = U::loadu(data_8_1);
    let mut zmm_2 = U::loadu(data_8_2);
    let mut zmm_3 = U::loadu(data_8_3);
    let mut zmm_4 = U::loadu(data_8_4);
    let mut zmm_5 = U::loadu(data_8_5);
    let mut zmm_6 = U::loadu(data_8_6);
    let mut zmm_7 = U::loadu(data_8_7);
    let mut zmm_8 = U::loadu(data_8_8);
    let mut zmm_9 = U::loadu(data_8_9);
    let mut zmm_10 = U::loadu(data_8_10);
    let mut zmm_11 = U::loadu(data_8_11);
    let mut zmm_12 = U::loadu(data_8_12);
    let mut zmm_13 = U::loadu(data_8_13);
    let mut zmm_14 = U::loadu(data_8_14);
    let mut zmm_15 = U::loadu(data_8_15);
    zmm_0 = sort_zmm_64bit(zmm_0);
    zmm_1 = sort_zmm_64bit(zmm_1);
    zmm_2 = sort_zmm_64bit(zmm_2);
    zmm_3 = sort_zmm_64bit(zmm_3);
    zmm_4 = sort_zmm_64bit(zmm_4);
    zmm_5 = sort_zmm_64bit(zmm_5);
    zmm_6 = sort_zmm_64bit(zmm_6);
    zmm_7 = sort_zmm_64bit(zmm_7);
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
    let mut zmm_4_2 = [zmm_4, zmm_5, zmm_6, zmm_7];
    let mut zmm_4_3 = [zmm_8, zmm_9, zmm_10, zmm_11];
    let mut zmm_4_4 = [zmm_12, zmm_13, zmm_14, zmm_15];

    bitonic_merge_four_zmm_64bit(&mut zmm_4_1);
    bitonic_merge_four_zmm_64bit(&mut zmm_4_2);
    bitonic_merge_four_zmm_64bit(&mut zmm_4_3);
    bitonic_merge_four_zmm_64bit(&mut zmm_4_4);

    let mut zmm_8_1 = [
        zmm_4_1[0], zmm_4_1[1], zmm_4_1[2], zmm_4_1[3], zmm_4_2[0], zmm_4_2[1], zmm_4_2[2],
        zmm_4_2[3],
    ];

    let mut zmm_8_2 = [
        zmm_4_3[0], zmm_4_3[1], zmm_4_3[2], zmm_4_3[3], zmm_4_4[0], zmm_4_4[1], zmm_4_4[2],
        zmm_4_4[3],
    ];

    bitonic_merge_eight_zmm_64bit(&mut zmm_8_1);
    bitonic_merge_eight_zmm_64bit(&mut zmm_8_2);

    let mut zmm_16_1 = [
        zmm_8_1[0], zmm_8_1[1], zmm_8_1[2], zmm_8_1[3], zmm_8_1[4], zmm_8_1[5], zmm_8_1[6],
        zmm_8_1[7], zmm_8_2[0], zmm_8_2[1], zmm_8_2[2], zmm_8_2[3], zmm_8_2[4], zmm_8_2[5],
        zmm_8_2[6], zmm_8_2[7],
    ];

    bitonic_merge_sixteen_zmm_64bit(&mut zmm_16_1);

    let ([mut zmm_4_5, mut zmm_4_6, mut zmm_4_7, mut zmm_4_8], data_splits) = {
        if n >= 192 {
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_0, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_1, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_2, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_3, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_4, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_5, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_6, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_7, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_8, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_9, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_10, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_11, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_12, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_13, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_14, data_8_15) = data_128_1.split_at_mut(split_index);

            let mut zmm_0 = U::mask_loadu(data_8_0);
            let mut zmm_1 = U::mask_loadu(data_8_1);
            let mut zmm_2 = U::mask_loadu(data_8_2);
            let mut zmm_3 = U::mask_loadu(data_8_3);
            let mut zmm_4 = U::mask_loadu(data_8_4);
            let mut zmm_5 = U::mask_loadu(data_8_5);
            let mut zmm_6 = U::mask_loadu(data_8_6);
            let mut zmm_7 = U::mask_loadu(data_8_7);
            let mut zmm_8 = U::mask_loadu(data_8_8);
            let mut zmm_9 = U::mask_loadu(data_8_9);
            let mut zmm_10 = U::mask_loadu(data_8_10);
            let mut zmm_11 = U::mask_loadu(data_8_11);
            let mut zmm_12 = U::mask_loadu(data_8_12);
            let mut zmm_13 = U::mask_loadu(data_8_13);
            let mut zmm_14 = U::mask_loadu(data_8_14);
            let mut zmm_15 = U::mask_loadu(data_8_15);

            zmm_0 = sort_zmm_64bit(zmm_0);
            zmm_1 = sort_zmm_64bit(zmm_1);
            zmm_2 = sort_zmm_64bit(zmm_2);
            zmm_3 = sort_zmm_64bit(zmm_3);
            zmm_4 = sort_zmm_64bit(zmm_4);
            zmm_5 = sort_zmm_64bit(zmm_5);
            zmm_6 = sort_zmm_64bit(zmm_6);
            zmm_7 = sort_zmm_64bit(zmm_7);
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

            (
                [
                    [zmm_0, zmm_1, zmm_2, zmm_3],
                    [zmm_4, zmm_5, zmm_6, zmm_7],
                    [zmm_8, zmm_9, zmm_10, zmm_11],
                    [zmm_12, zmm_13, zmm_14, zmm_15],
                ],
                [
                    data_8_0, data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7,
                    data_8_8, data_8_9, data_8_10, data_8_11, data_8_12, data_8_13, data_8_14,
                    data_8_15,
                ],
            )
        } else {
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_0, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_1, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_2, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_3, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_4, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_5, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_6, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_7, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_8, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_9, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_10, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_11, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_12, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_13, data_128_1) = data_128_1.split_at_mut(split_index);
            let split_index = cmp::min(data_128_1.len(), 8);
            let (data_8_14, data_8_15) = data_128_1.split_at_mut(split_index);
            let mut zmm_0 = U::mask_loadu(data_8_0);
            let mut zmm_1 = U::mask_loadu(data_8_1);
            let mut zmm_2 = U::mask_loadu(data_8_2);
            let mut zmm_3 = U::mask_loadu(data_8_3);
            let mut zmm_4 = U::mask_loadu(data_8_4);
            let mut zmm_5 = U::mask_loadu(data_8_5);
            let mut zmm_6 = U::mask_loadu(data_8_6);
            let mut zmm_7 = U::mask_loadu(data_8_7);

            zmm_0 = sort_zmm_64bit(zmm_0);
            zmm_1 = sort_zmm_64bit(zmm_1);
            zmm_2 = sort_zmm_64bit(zmm_2);
            zmm_3 = sort_zmm_64bit(zmm_3);
            zmm_4 = sort_zmm_64bit(zmm_4);
            zmm_5 = sort_zmm_64bit(zmm_5);
            zmm_6 = sort_zmm_64bit(zmm_6);
            zmm_7 = sort_zmm_64bit(zmm_7);

            let mut zmm_8 = U::set(T::MAX_VALUE);
            let mut zmm_9 = U::set(T::MAX_VALUE);
            let mut zmm_10 = U::set(T::MAX_VALUE);
            let mut zmm_11 = U::set(T::MAX_VALUE);
            let mut zmm_12 = U::set(T::MAX_VALUE);
            let mut zmm_13 = U::set(T::MAX_VALUE);
            let mut zmm_14 = U::set(T::MAX_VALUE);
            let mut zmm_15 = U::set(T::MAX_VALUE);

            bitonic_merge_two_zmm_64bit(&mut zmm_0, &mut zmm_1);
            bitonic_merge_two_zmm_64bit(&mut zmm_2, &mut zmm_3);
            bitonic_merge_two_zmm_64bit(&mut zmm_4, &mut zmm_5);
            bitonic_merge_two_zmm_64bit(&mut zmm_6, &mut zmm_7);
            bitonic_merge_two_zmm_64bit(&mut zmm_8, &mut zmm_9);
            bitonic_merge_two_zmm_64bit(&mut zmm_10, &mut zmm_11);
            bitonic_merge_two_zmm_64bit(&mut zmm_12, &mut zmm_13);
            bitonic_merge_two_zmm_64bit(&mut zmm_14, &mut zmm_15);
            (
                [
                    [zmm_0, zmm_1, zmm_2, zmm_3],
                    [zmm_4, zmm_5, zmm_6, zmm_7],
                    [zmm_8, zmm_9, zmm_10, zmm_11],
                    [zmm_12, zmm_13, zmm_14, zmm_15],
                ],
                [
                    data_8_0, data_8_1, data_8_2, data_8_3, data_8_4, data_8_5, data_8_6, data_8_7,
                    data_8_8, data_8_9, data_8_10, data_8_11, data_8_12, data_8_13, data_8_14,
                    data_8_15,
                ],
            )
        }
    };

    bitonic_merge_four_zmm_64bit(&mut zmm_4_5);
    bitonic_merge_four_zmm_64bit(&mut zmm_4_6);
    bitonic_merge_four_zmm_64bit(&mut zmm_4_7);
    bitonic_merge_four_zmm_64bit(&mut zmm_4_8);

    let mut zmm_8_3 = [
        zmm_4_5[0], zmm_4_5[1], zmm_4_5[2], zmm_4_5[3], zmm_4_6[0], zmm_4_6[1], zmm_4_6[2],
        zmm_4_6[3],
    ];

    let mut zmm_8_4 = [
        zmm_4_7[0], zmm_4_7[1], zmm_4_7[2], zmm_4_7[3], zmm_4_8[0], zmm_4_8[1], zmm_4_8[2],
        zmm_4_8[3],
    ];

    bitonic_merge_eight_zmm_64bit(&mut zmm_8_3);
    bitonic_merge_eight_zmm_64bit(&mut zmm_8_4);

    let mut zmm_16_2 = [
        zmm_8_3[0], zmm_8_3[1], zmm_8_3[2], zmm_8_3[3], zmm_8_3[4], zmm_8_3[5], zmm_8_3[6],
        zmm_8_3[7], zmm_8_4[0], zmm_8_4[1], zmm_8_4[2], zmm_8_4[3], zmm_8_4[4], zmm_8_4[5],
        zmm_8_4[6], zmm_8_4[7],
    ];
    bitonic_merge_sixteen_zmm_64bit(&mut zmm_16_2);

    let mut zmm = [
        zmm_16_1[0],
        zmm_16_1[1],
        zmm_16_1[2],
        zmm_16_1[3],
        zmm_16_1[4],
        zmm_16_1[5],
        zmm_16_1[6],
        zmm_16_1[7],
        zmm_16_1[8],
        zmm_16_1[9],
        zmm_16_1[10],
        zmm_16_1[11],
        zmm_16_1[12],
        zmm_16_1[13],
        zmm_16_1[14],
        zmm_16_1[15],
        zmm_16_2[0],
        zmm_16_2[1],
        zmm_16_2[2],
        zmm_16_2[3],
        zmm_16_2[4],
        zmm_16_2[5],
        zmm_16_2[6],
        zmm_16_2[7],
        zmm_16_2[8],
        zmm_16_2[9],
        zmm_16_2[10],
        zmm_16_2[11],
        zmm_16_2[12],
        zmm_16_2[13],
        zmm_16_2[14],
        zmm_16_2[15],
    ];

    bitonic_merge_thirtytwo_zmm_64bit(&mut zmm);

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

    U::mask_storeu(zmm[16], data_splits[0]);
    U::mask_storeu(zmm[17], data_splits[1]);
    U::mask_storeu(zmm[18], data_splits[2]);
    U::mask_storeu(zmm[19], data_splits[3]);
    U::mask_storeu(zmm[20], data_splits[4]);
    U::mask_storeu(zmm[21], data_splits[5]);
    U::mask_storeu(zmm[22], data_splits[6]);
    U::mask_storeu(zmm[23], data_splits[7]);
    if n >= 192 {
        U::mask_storeu(zmm[24], data_splits[8]);
        U::mask_storeu(zmm[25], data_splits[9]);
        U::mask_storeu(zmm[26], data_splits[10]);
        U::mask_storeu(zmm[27], data_splits[11]);
        U::mask_storeu(zmm[28], data_splits[12]);
        U::mask_storeu(zmm[29], data_splits[13]);
        U::mask_storeu(zmm[30], data_splits[14]);
        U::mask_storeu(zmm[31], data_splits[15]);
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
        data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= 128
     */
    if data.len() <= 256 {
        sort_256::<T, U>(data);
        return;
    }

    let pivot = get_pivot_64bit::<T, U>(data);
    let mut smallest = T::MAX_VALUE;
    let mut biggest = T::MIN_VALUE;
    //let pivot_index = partition_avx512::<T, 8, U>(data, pivot, &mut smallest, &mut biggest);
    let pivot_index =
        partition_avx512_unrolled::<T, 8, U, 8>(data, pivot, &mut smallest, &mut biggest);
    let (left, right) = data.split_at_mut(pivot_index);
    if pivot != smallest {
        qsort_64bit_::<T, U>(left, max_iters - 1);
    }
    if pivot != biggest {
        qsort_64bit_::<T, U>(right, max_iters - 1);
    }
}

#[cfg(test)]
pub(crate) mod test {
    macro_rules! test_min_max {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<test_min_max_ $ty>]() {
                    let first = $simd::loadu(&[
                        1 as $ty,
                        20 as $ty,
                        3 as $ty ,
                        40 as $ty,
                        5 as $ty,
                        60 as $ty,
                        70 as $ty,
                        80 as $ty]);
                    let second = $simd::loadu(&[
                        10 as $ty,
                        2 as $ty,
                        30 as $ty,
                        4 as $ty,
                        50 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8 as $ty]);
                    assert_eq!(
                        $into_array(<$simd as SimdCompare<$ty, 8>>::min(first, second)),
                        [
                            1 as $ty,
                            2 as $ty,
                            3 as $ty,
                            4 as $ty,
                            5 as $ty,
                            6 as $ty,
                            7 as $ty,
                            8 as $ty,
                        ]
                    );
                    assert_eq!(
                        $into_array(<$simd as SimdCompare<$ty, 8>>::max(first, second)),
                        [
                            10 as $ty,
                            20 as $ty,
                            30 as $ty,
                            40 as $ty,
                            50 as $ty,
                            60 as $ty,
                            70 as $ty,
                            80 as $ty,
                        ]
                    );
                }
            }
        };
    }

    macro_rules! test_loadu_storeu {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<test_loadu_storeu_ $ty>]() {
                    let mut input_slice = [
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8 as $ty,
                        9 as $ty,
                        10 as $ty,
                    ];
                    let first = $simd::loadu(input_slice.as_ref());
                    assert_eq!($into_array(first), [
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8 as $ty,
                    ]);
                    $simd::storeu(first, &mut input_slice[2..]);
                    assert_eq!(input_slice, [
                        1 as $ty,
                        2 as $ty,
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8 as $ty,
                    ]);
                }
            }
        };
    }

    macro_rules! test_mask_loadu_mask_storeu {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<test_mask_loadu_mask_storeu_ $ty>]() {
                    let mut input_slice = [
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8 as $ty,
                        9 as $ty,
                        10  as $ty,
                    ];
                    let first = $simd::mask_loadu(&input_slice[..2]);
                    assert_eq!(
                        $into_array(first),
                        [
                            1 as $ty,
                            2 as $ty,
                            $ty::MAX,
                            $ty::MAX,
                            $ty::MAX,
                            $ty::MAX,
                            $ty::MAX,
                            $ty::MAX
                        ]
                    );
                    $simd::mask_storeu(first, &mut input_slice[2..4]);
                    assert_eq!(input_slice, [
                        1 as $ty,
                        2 as $ty,
                        1 as $ty,
                        2 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8 as $ty,
                        9 as $ty,
                        10 as $ty,
                    ]);
                }
            }
        };
    }

    macro_rules! test_get_at_index {
        ($ty: ident, $simd: ident) => {
            paste::paste! {
                #[test]
                fn [<test_get_at_index_ $ty>]() {
                    let first = $simd::mask_loadu(&[
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8 as $ty,
                    ]);
                    for i in 1..9 {
                        assert_eq!(i as $ty, $simd::get_value_at_idx(first, i - 1));
                    }
                }
            }
        };
    }

    macro_rules! test_ge {
        ($ty: ident, $simd: ident, $mask_result: expr) => {
            paste::paste! {
                #[test]
                fn [<test_ge_ $ty>]() {
                    let first = $simd::mask_loadu(&[
                        1 as $ty, 20 as $ty,
                        3 as $ty,
                        40 as $ty,
                        5 as $ty,
                        60 as $ty,
                        7 as $ty,
                        80 as $ty
                    ]);
                    let second = $simd::mask_loadu(&[
                        10 as $ty,
                        2 as $ty,
                        30 as $ty,
                        40 as $ty,
                        50 as $ty,
                        6 as $ty,
                        70 as $ty,
                        80 as $ty
                    ]);
                    let result_mask = <$simd as SimdCompare<$ty, 8>>::ge(first, second);
                    assert_eq!(result_mask, $mask_result);
                }
            }
        };
    }

    macro_rules! test_gather {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<test_gather_ $ty>]() {
                    let input_slice = [
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8 as $ty,
                        9 as $ty,
                        10 as $ty,
                    ];
                    let first = $simd::gather_from_idx([1, 1, 2, 2, 9, 9, 5, 6], input_slice.as_ref());
                    assert_eq!($into_array(first), [
                        2 as $ty,
                        2 as $ty,
                        3 as $ty,
                        3 as $ty,
                        10 as $ty,
                        10 as $ty,
                        6 as $ty,
                        7 as $ty,
                    ]);
                }
            }
        };
    }

    macro_rules! test_not {
        ($ty: ident, $simd: ident, $mask_input: expr, $mask_result: expr) => {
            paste::paste! {
                #[test]
                fn [<test_not_ $ty>]() {
                    let first: <$simd as SimdCompare<$ty, 8>>::OPMask = $mask_input;
                    assert_eq!(<$simd as SimdCompare<$ty, 8>>::not_mask(first), $mask_result);
                }
            }
        };
    }

    macro_rules! test_count_ones {
        ($ty: ident, $simd: ident, $mask_fn: ident) => {
            paste::paste! {
                #[test]
                fn [<test_count_ones_ $ty>]() {
                    for i in 0u8..8 {
                        let mask = $mask_fn(i);
                        assert_eq!(<$simd as SimdCompare<$ty, 8>>::ones_count(mask), i.count_ones() as usize);
                    }
                }
            }
        };
    }

    macro_rules! test_reduce_min_max {
        ($ty: ident, $simd: ident) => {
            paste::paste! {
                #[test]
                fn [<test_reduce_min_max_ $ty>]() {
                    let first = $simd::mask_loadu(&[
                        1 as $ty,
                        6 as $ty,
                        3 as $ty,
                        4 as $ty,
                        1 as $ty,
                        2 as $ty,
                        9 as $ty,
                        8 as $ty
                    ]);
                    assert_eq!(<$simd as SimdCompare<$ty, 8>>::reducemin(first), 1 as $ty);
                    assert_eq!(<$simd as SimdCompare<$ty, 8>>::reducemax(first), 9 as $ty);
                }
            }
        };
    }

    macro_rules! test_compress_store_u {
        ($ty: ident, $simd: ident, $mask_ty: ident,$generate_fn: ident) => {
            paste::paste! {
                #[test]
                fn [<test_compress_store_u_ $ty>]() {
                    let input_slice = [
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8 as $ty,
                        9 as $ty,
                        10 as $ty];
                    let first = $simd::loadu(input_slice.as_ref());
                    for i in 0..255 {
                        let (mask, new_values) = $generate_fn::<$ty, $mask_ty>(i, &input_slice);
                        format!("{:?}", mask);
                        let mut new_array = input_slice.clone();
                        $simd::mask_compressstoreu(&mut new_array[2..], mask, first);
                        format!("{:?}", new_array);
                        format!("{:?}", new_values);
                        println!("{:?}", new_array);
                        for j in 0..(i as usize).count_ones() as usize {
                            assert_eq!(new_array[2 + j], new_values[j]);
                        }
                        for j in i as usize..8 {
                            assert_eq!(new_array[2 + j], input_slice[2 + j]);
                        }
                    }
                }
            }
        };
    }

    macro_rules! test_shuffle1_1_1_1 {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<test_shuffle1_1_1_1_ $ty>]() {
                    let first = $simd::loadu(&[
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8  as $ty
                    ]);
                    assert_eq!(
                        $into_array(<$simd as Bit64Simd<$ty>>::shuffle1_1_1_1(first)),
                        [
                            2 as $ty,
                            1 as $ty,
                            4 as $ty,
                            3 as $ty,
                            6 as $ty,
                            5 as $ty,
                            8 as $ty,
                            7 as $ty,
                        ]
                    );
                }
            }
        };
    }

    macro_rules! test_swizzle2_0xaa {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<test_swizzle2_0xaa_ $ty>]() {
                    let first = $simd::loadu(&[
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8 as $ty,
                    ]);
                    let second = $simd::loadu(&[
                        10 as $ty,
                        20 as $ty,
                        30 as $ty,
                        40 as $ty,
                        50 as $ty,
                        60 as $ty,
                        70 as $ty,
                        80 as $ty
                    ]);
                    assert_eq!(
                        $into_array(<$simd as Bit64Simd<$ty>>::swizzle2_0xaa(first, second)),
                        [
                            1 as $ty,
                            20 as $ty,
                            3 as $ty,
                            40 as $ty,
                            5 as $ty,
                            60 as $ty,
                            7 as $ty,
                            80 as $ty
                        ]
                    );
                }
            }
        };
    }

    macro_rules! test_swizzle2_0xcc {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<test_swizzle2_0xcc_ $ty>]() {
                    let first = $simd::loadu(&[
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8  as $ty
                    ]);
                    let second = $simd::loadu(&[
                        10 as $ty,
                        20 as $ty,
                        30 as $ty,
                        40 as $ty,
                        50 as $ty,
                        60 as $ty,
                        70 as $ty,
                        80 as $ty
                    ]);
                    assert_eq!(
                        $into_array(<$simd as Bit64Simd<$ty>>::swizzle2_0xcc(first, second)),
                        [
                            1 as $ty,
                            2 as $ty,
                            30 as $ty,
                            40 as $ty,
                            5 as $ty,
                            6 as $ty,
                            70 as $ty,
                            80 as $ty

                       ]
                    );
                }
            }
        };
    }

    macro_rules! test_swizzle2_0xf0 {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<test_swizzle2_0xf0_ $ty>]() {
                    let first = $simd::loadu(&[
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty,
                        8  as $ty
                    ]);
                    let second = $simd::loadu(&[
                        10 as $ty,
                        20 as $ty,
                        30 as $ty,
                        40 as $ty,
                        50 as $ty,
                        60 as $ty,
                        70 as $ty,
                        80 as $ty
                    ]);
                    assert_eq!(
                        $into_array(<$simd as Bit64Simd<$ty>>::swizzle2_0xf0(first, second)),
                        [
                            1 as $ty,
                            2 as $ty,
                            3 as $ty,
                            4 as $ty,
                            50 as $ty,
                            60 as $ty,
                            70 as $ty,
                            80 as $ty,
                        ]
                    );
                }
            }
        };
    }

    macro_rules! network64bit1 {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<network64bit1_ $ty>]() {
                    let first = $simd::loadu(&[
                        0 as $ty,
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty
                    ]);
                    assert_eq!(
                        $into_array(<$simd as Bit64Simd<$ty>>::network64bit1(first)),
                        [
                            3 as $ty,
                            2 as $ty,
                            1 as $ty,
                            0 as $ty,
                            7 as $ty,
                            6 as $ty,
                            5 as $ty,
                            4 as $ty,
                        ]
                    );
                }
            }
        };
    }

    macro_rules! network64bit2 {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<network64bit2_ $ty>]() {
                    let first = $simd::loadu(&[
                        0 as $ty,
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty
                    ]);
                    assert_eq!(
                        $into_array(<$simd as Bit64Simd<$ty>>::network64bit2(first)),
                        [
                            7 as $ty,
                            6 as $ty,
                            5 as $ty,
                            4 as $ty,
                            3 as $ty,
                            2 as $ty,
                            1 as $ty,
                            0 as $ty
                        ]
                    );
                }
            }
        };
    }

    macro_rules! network64bit3 {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<network64bit3_ $ty>]() {
                    let first = $simd::loadu(&[
                        0 as $ty,
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty
                    ]);
                    assert_eq!(
                        $into_array(<$simd as Bit64Simd<$ty>>::network64bit3(first)),
                        [
                            2 as $ty,
                            3 as $ty,
                            0 as $ty,
                            1 as $ty,
                            6 as $ty,
                            7 as $ty,
                            4 as $ty,
                            5 as $ty
                        ]
                    );
                }
            }
        };
    }

    macro_rules! network64bit4 {
        ($ty: ident, $simd: ident, $into_array: ident) => {
            paste::paste! {
                #[test]
                fn [<network64bit4_ $ty>]() {
                    let first = $simd::loadu(&[
                        0 as $ty,
                        1 as $ty,
                        2 as $ty,
                        3 as $ty,
                        4 as $ty,
                        5 as $ty,
                        6 as $ty,
                        7 as $ty
                    ]);
                    assert_eq!(
                        $into_array(<$simd as Bit64Simd<$ty>>::network64bit4(first)),
                        [
                            4 as $ty,
                            5 as $ty,
                            6 as $ty,
                            7 as $ty,
                            0 as $ty,
                            1 as $ty,
                            2 as $ty,
                            3 as $ty,
                        ]
                    );
                }
            }
        };
    }

    macro_rules! test_sort_n {
        ($ty: ident, $simd: ident, $n: literal) => {
            paste::paste! {
                #[test]
                fn [<test_sort_ $n _ $ty >]() {
                    let result: Vec<$ty> = (0..$n).into_iter().map(|x| x as $ty).collect();
                    for i in 0..$n {
                        let mut array = Vec::with_capacity(i);
                        array.extend_from_slice(&result[..i]);
                        array.reverse();
                        [<sort_ $n>]::<$ty, $simd>(&mut array);
                        assert_eq!(&array, &result[..i]);
                    }
                }
            }
        };
    }

    macro_rules! test_sort_e2e {
        ($ty: ident, $simd: ident, $sort: ident) => {
            paste::paste! {
                #[test]
                fn [<test_sort_e2e_ $ty >]() {
                    let start = 0;
                    let end = 1024;
                    let result: Vec<$ty> = (0..end).into_iter().map(|x| x as $ty).collect();
                    for i in start as usize..end as usize {
                        let mut array = Vec::with_capacity(i);
                        array.extend_from_slice(&result[..i]);
                        array.reverse();
                        $sort(array.as_mut_slice());
                        assert_eq!(&array, &result[..i]);
                        println!("succeeded {}", i);
                    }
                }
            }
        };
    }

    pub(crate) use {
        network64bit1, network64bit2, network64bit3, network64bit4, test_compress_store_u,
        test_count_ones, test_gather, test_ge, test_get_at_index, test_loadu_storeu,
        test_mask_loadu_mask_storeu, test_min_max, test_not, test_reduce_min_max,
        test_shuffle1_1_1_1, test_sort_e2e, test_sort_n, test_swizzle2_0xaa, test_swizzle2_0xcc,
        test_swizzle2_0xf0,
    };
}
