#![cfg_attr(feature = "nightly", feature(portable_simd))]
#![cfg_attr(feature = "nightly", feature(stdsimd))]

pub(crate) mod bit_64;
pub mod platform;

use std::{
    cmp::{max_by, min_by, Ordering},
    fmt::Debug,
    mem::{self, MaybeUninit},
};

#[cfg(feature = "nightly")]
use std::simd::SimdElement;

pub trait SimdCompare<T: SimdSortable, const NUM_LANES: usize>: Copy + Debug {
    type OPMask: Copy + Debug;

    fn min(a: Self, b: Self) -> Self;
    fn max(a: Self, b: Self) -> Self;
    #[inline]
    fn mask_mov(a: Self, b: Self, mask: fn(Self, Self) -> Self) -> Self {
        mask(a, b)
    }
    #[inline]
    fn shuffle(a: Self, shuffle_fn: fn(Self) -> Self) -> Self {
        shuffle_fn(a)
    }
    fn loadu(data: &[T]) -> Self;
    fn storeu(input: Self, data: &mut [T]);

    fn mask_loadu(data: &[T]) -> Self;
    fn mask_storeu(input: Self, data: &mut [T]);

    fn gather_from_idx(idx: [usize; NUM_LANES], data: &[T]) -> Self;

    fn get_value_at_idx(input: Self, idx: usize) -> T;

    fn set(value: T) -> Self;

    fn ge(a: Self, b: Self) -> Self::OPMask;

    fn ones_count(mask: Self::OPMask) -> usize;
    fn not_mask(mask: Self::OPMask) -> Self::OPMask;

    fn reducemin(x: Self) -> T;
    fn reducemax(x: Self) -> T;

    fn mask_compressstoreu(array: &mut [T], mask: Self::OPMask, data: Self);
}

#[cfg(feature = "nightly")]
pub trait SimdSortable: PartialOrd + SimdElement + Debug {
    const MAX_VALUE: Self;
    const MIN_VALUE: Self;
}

#[cfg(not(feature = "nightly"))]
pub trait SimdSortable: PartialOrd + Copy + Debug {
    const MAX_VALUE: Self;
    const MIN_VALUE: Self;
}

impl SimdSortable for u64 {
    const MAX_VALUE: Self = u64::MAX;
    const MIN_VALUE: Self = u64::MIN;
}

impl SimdSortable for i64 {
    const MAX_VALUE: Self = i64::MAX;
    const MIN_VALUE: Self = i64::MIN;
}

impl SimdSortable for f64 {
    const MAX_VALUE: Self = f64::MAX;
    const MIN_VALUE: Self = f64::MIN;
}

pub(crate) fn comparison_func<T: SimdSortable>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b).unwrap()
}

///
/// COEX == Compare and Exchange two registers by swapping min and max values
///
#[inline]
pub(crate) fn coex<T: SimdSortable, const N: usize, U: SimdCompare<T, N>>(a: &mut U, b: &mut U) {
    let temp = *a;
    *a = U::min(*a, *b);
    *b = U::max(temp, *b);
}

#[inline]
pub(crate) fn cmp_merge<T: SimdSortable, const N: usize, U: SimdCompare<T, N>>(
    in1: U,
    in2: U,
    mask: fn(U, U) -> U,
) -> U {
    let min_values = U::min(in2, in1);
    let max_values = U::max(in2, in1);
    U::mask_mov(min_values, max_values, mask) // 0 -> min, 1 -> max
}

/*
 * Parition one ZMM register based on the pivot and returns the index of the
 * last element that is less than equal to the pivot.
 */
fn partition_vec<T: SimdSortable, const N: usize, U: SimdCompare<T, N>>(
    data: &mut [T],
    left: usize,
    right: usize,
    curr_vec: &U,
    pivot_vec: &U,
    smallest_vec: &mut U,
    biggest_vec: &mut U,
) -> usize {
    /* which elements are larger than the pivot */
    let gt_mask = U::ge(*curr_vec, *pivot_vec);
    let amount_gt_pivot = U::ones_count(gt_mask);
    // This is safe since we are accessing elements within bounds
    // get_unchecked call is used to get rid of unwanted bounds check
    unsafe {
        U::mask_compressstoreu(
            data.get_unchecked_mut(left..),
            U::not_mask(gt_mask),
            *curr_vec,
        );
        U::mask_compressstoreu(
            data.get_unchecked_mut((right - amount_gt_pivot)..),
            gt_mask,
            *curr_vec,
        );
    }
    *smallest_vec = U::min(*curr_vec, *smallest_vec);
    *biggest_vec = U::max(*curr_vec, *biggest_vec);
    return amount_gt_pivot;
}

#[inline]
pub(crate) fn partition_avx512<T: SimdSortable, const N: usize, U: SimdCompare<T, N>>(
    data: &mut [T],
    pivot: T,
    smallest: &mut T,
    biggest: &mut T,
) -> usize {
    /* make array length divisible by N , shortening the array */
    let mut left = 0;
    let mut right = data.len();
    let mut i = (right - left) % N;
    while i > 0 {
        *smallest = min_by(*smallest, data[left], comparison_func);
        *biggest = max_by(*biggest, data[left], comparison_func);
        if comparison_func(&data[left], &pivot) != Ordering::Less {
            right -= 1;
            data.swap(left, right);
        } else {
            left += 1;
        }
        i -= 1;
    }

    if left == right {
        return left; /* less than N elements in the array */
    }

    let pivot_vec = U::set(pivot);
    let mut min_vec = U::set(*smallest);
    let mut max_vec = U::set(*biggest);

    if right - left == N {
        // This is safe since we are in bounds
        // get_unchecked call is used to get rid of bound checks
        let vec_ = unsafe { U::loadu(data.get_unchecked(left..)) };
        let amount_gt_pivot = partition_vec(
            data,
            left,
            left + N,
            &vec_,
            &pivot_vec,
            &mut min_vec,
            &mut max_vec,
        );
        *smallest = U::reducemin(min_vec);
        *biggest = U::reducemax(max_vec);
        return left + (N - amount_gt_pivot);
    }

    // first and last N values are partitioned at the end
    let vec_left = U::loadu(&data[left..]);
    let vec_right = U::loadu(&data[(right - N)..]);
    // store points of the vectors
    let mut r_store = right - N;
    let mut l_store = left;
    // indices for loading the elements
    left += N;
    right -= N;
    while right - left != 0 {
        let curr_vec;
        /*
         * if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side
         */
        if (r_store + N) - right < left - l_store {
            right -= N;
            curr_vec = U::loadu(&data[right..]);
        } else {
            curr_vec = U::loadu(&data[left..]);
            left += N;
        }
        // partition the current vector and save it on both sides of the array
        let amount_gt_pivot = partition_vec(
            data,
            l_store,
            r_store + N,
            &curr_vec,
            &pivot_vec,
            &mut min_vec,
            &mut max_vec,
        );
        r_store -= amount_gt_pivot;
        l_store += N - amount_gt_pivot;
    }

    /* partition and save vec_left and vec_right */
    let mut amount_gt_pivot = partition_vec(
        data,
        l_store,
        r_store + N,
        &vec_left,
        &pivot_vec,
        &mut min_vec,
        &mut max_vec,
    );
    l_store += N - amount_gt_pivot;
    amount_gt_pivot = partition_vec(
        data,
        l_store,
        l_store + N,
        &vec_right,
        &pivot_vec,
        &mut min_vec,
        &mut max_vec,
    );
    l_store += N - amount_gt_pivot;
    *smallest = U::reducemin(min_vec);
    *biggest = U::reducemax(max_vec);
    return l_store;
}

#[inline]
pub(crate) fn partition_avx512_unrolled<T, const N: usize, U, const UNROLL: usize>(
    data: &mut [T],
    pivot: T,
    smallest: &mut T,
    biggest: &mut T,
) -> usize
where
    T: SimdSortable,
    U: SimdCompare<T, N>,
{
    let mut left = 0;
    let mut right = data.len();
    if right - left <= 2 * UNROLL * N {
        return partition_avx512::<T, N, U>(data, pivot, smallest, biggest);
    }
    /* make array length divisible by 8*vtype::numlanes , shortening the array */

    let mut i = (right - left) % (UNROLL * N);
    while i > 0 {
        // This is safe since left is in bounds
        let other = unsafe { data.get_unchecked(left) };
        *smallest = min_by(*smallest, *other, comparison_func);
        *biggest = max_by(*biggest, *other, comparison_func);
        if comparison_func(other, &pivot) != Ordering::Less {
            right -= 1;
            data.swap(left, right);
        } else {
            left += 1;
        }
        i -= 1;
    }

    if left == right {
        return left; /* less than N elements in the array */
    }

    let pivot_vec = U::set(pivot);
    let mut min_vec = U::set(*smallest);
    let mut max_vec = U::set(*biggest);

    // We will now have atleast 16 registers worth of data to process:
    // left and right vtype::numlanes values are partitioned at the end

    let (vec_left, vec_right) = {
        // This is safe since we are loading data within bounds verified by the first if
        // get_unchecked call is used to get rid of bound checks
        unsafe {
            let mut vec_left: [MaybeUninit<U>; UNROLL] = MaybeUninit::uninit().assume_init();
            let mut vec_right: [MaybeUninit<U>; UNROLL] = MaybeUninit::uninit().assume_init();
            for i in 0..UNROLL {
                vec_left[i] = MaybeUninit::new(U::loadu(data.get_unchecked((left + N * i)..)));
                vec_right[i] =
                    MaybeUninit::new(U::loadu(data.get_unchecked((right - N * (UNROLL - i))..)));
            }

            (
                mem::transmute_copy::<_, [U; UNROLL]>(&vec_left),
                mem::transmute_copy::<_, [U; UNROLL]>(&vec_right),
            )
        }
    };

    // store points of the vectors
    let mut r_store = right - N;
    let mut l_store = left;

    left += N * UNROLL;
    right -= N * UNROLL;

    while right - left != 0 {
        // This is safe since left and right are in bounds
        // get_unchecked call is used to get rid of bound checks
        let current_vec = unsafe {
            let mut current_vec: [MaybeUninit<U>; UNROLL] = MaybeUninit::uninit().assume_init();

            /*
             * if fewer elements are stored on the right side of the array,
             * then next elements are loaded from the right side,
             * otherwise from the left side
             */
            if (r_store + N) - right < left - l_store {
                right -= UNROLL * N;
                for i in 0..UNROLL {
                    current_vec[i] =
                        MaybeUninit::new(U::loadu(data.get_unchecked((right + (N * i))..)));
                }
            } else {
                for i in 0..UNROLL {
                    current_vec[i] =
                        MaybeUninit::new(U::loadu(data.get_unchecked((left + (N * i))..)));
                }
                left += UNROLL * N;
            }

            mem::transmute_copy::<_, [U; UNROLL]>(&current_vec)
        };

        // partition the current vector and save it on both sides of the array
        for i in 0..UNROLL {
            let amount_ge_pivot = partition_vec(
                data,
                l_store,
                r_store + N,
                &current_vec[i],
                &pivot_vec,
                &mut min_vec,
                &mut max_vec,
            );
            l_store += N - amount_ge_pivot;
            r_store -= amount_ge_pivot;
        }
    }

    //  partition and save vec_left[8] and vec_right[8]
    for i in 0..UNROLL {
        let amount_ge_pivot = partition_vec(
            data,
            l_store,
            r_store + N,
            &vec_left[i],
            &pivot_vec,
            &mut min_vec,
            &mut max_vec,
        );
        l_store += N - amount_ge_pivot;
        r_store -= amount_ge_pivot;
    }

    for i in 0..UNROLL {
        let amount_ge_pivot = partition_vec(
            data,
            l_store,
            r_store + N,
            &vec_right[i],
            &pivot_vec,
            &mut min_vec,
            &mut max_vec,
        );
        l_store += N - amount_ge_pivot;
        r_store -= amount_ge_pivot;
    }

    *smallest = U::reducemin(min_vec);
    *biggest = U::reducemax(max_vec);
    return l_store;
}
