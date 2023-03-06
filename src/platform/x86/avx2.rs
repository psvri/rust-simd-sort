use core::arch::x86_64::__m256i;
use core::arch::x86_64::_mm256_castps_si256;
use core::arch::x86_64::_mm256_castsi256_ps;
use core::arch::x86_64::_mm256_max_epi32;
use core::arch::x86_64::_mm256_min_epi32;
use core::arch::x86_64::_mm256_permutevar8x32_epi32;
use core::arch::x86_64::_mm256_setr_epi32;
use core::arch::x86_64::_mm256_shuffle_epi32;
use core::arch::x86_64::_mm256_shuffle_ps;
use std::arch::x86_64::_mm256_blend_epi32;
use std::arch::x86_64::_mm256_unpackhi_epi32;
use std::arch::x86_64::_mm256_unpacklo_epi32;

// will be removed once [#27731](https://doc.rust-lang.org/core/arch/x86/fn._MM_SHUFFLE.html)
// is stabalised
const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

macro_rules! COEX {
    ($a:expr, $b:expr) => {{
        let vec_tmp = $a;
        $a = _mm256_min_epi32($a, $b);
        $b = _mm256_max_epi32(vec_tmp, $b);
    }};
}

macro_rules! SHUFFLE_2_VECS {
    ($a:expr, $b:expr, $mask:expr) => {
        _mm256_castps_si256(_mm256_shuffle_ps(
            _mm256_castsi256_ps($a),
            _mm256_castsi256_ps($b),
            $mask,
        ))
    };
}

macro_rules! SORT_16 {
    ($v1:expr, $v2:expr) => {
        COEX!($v1, $v2); /* step 1 */

        $v2 = _mm256_shuffle_epi32($v2, _MM_SHUFFLE(2, 3, 0, 1)); /* step 2 */
        COEX!($v1, $v2);

        let mut tmp = $v1; /* step  3 */
        $v1 = SHUFFLE_2_VECS!($v1, $v2, 0b10001000);
        $v2 = SHUFFLE_2_VECS!(tmp, $v2, 0b11011101);
        COEX!($v1, $v2);

        $v2 = _mm256_shuffle_epi32($v2, _MM_SHUFFLE(0, 1, 2, 3)); /* step  4 */
        COEX!($v1, $v2);

        tmp = $v1; /* step  5 */
        $v1 = SHUFFLE_2_VECS!($v1, $v2, 0b01000100);
        $v2 = SHUFFLE_2_VECS!(tmp, $v2, 0b11101110);
        COEX!($v1, $v2);

        tmp = $v1; /* step  6 */
        $v1 = SHUFFLE_2_VECS!($v1, $v2, 0b11011000);
        $v2 = SHUFFLE_2_VECS!(tmp, $v2, 0b10001101);
        COEX!($v1, $v2);

        $v2 = _mm256_permutevar8x32_epi32($v2, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
        COEX!($v1, $v2); /* step  7 */

        tmp = $v1; /* step  8 */
        $v1 = SHUFFLE_2_VECS!($v1, $v2, 0b11011000);
        $v2 = SHUFFLE_2_VECS!(tmp, $v2, 0b10001101);
        COEX!($v1, $v2);

        tmp = $v1; /* step  9 */
        $v1 = SHUFFLE_2_VECS!($v1, $v2, 0b11011000);
        $v2 = SHUFFLE_2_VECS!(tmp, $v2, 0b10001101);
        COEX!($v1, $v2);

        /* permute to make it easier to restore order */
        $v1 = _mm256_permutevar8x32_epi32($v1, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));
        $v2 = _mm256_permutevar8x32_epi32($v2, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));

        tmp = $v1; /* step  10 */
        $v1 = SHUFFLE_2_VECS!($v1, $v2, 0b10001000);
        $v2 = SHUFFLE_2_VECS!(tmp, $v2, 0b11011101);
        COEX!($v1, $v2);

        /* restore order */
        let b2 = _mm256_shuffle_epi32($v2, 0b10110001);
        let b1 = _mm256_shuffle_epi32($v1, 0b10110001);
        $v1 = _mm256_blend_epi32($v1, b2, 0b10101010);
        $v2 = _mm256_blend_epi32(b1, $v2, 0b10101010);
    };
}

macro_rules! ASC {
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr) => {
        ((($h < 7) as i32) << 7)
            | ((($g < 6) as i32) << 6)
            | ((($f < 5) as i32) << 5)
            | ((($e < 4) as i32) << 4)
            | ((($d < 3) as i32) << 3)
            | ((($c < 2) as i32) << 2)
            | ((($b < 1) as i32) << 1)
            | (($a < 0) as i32)
    };
}

macro_rules! COEX_PERMUTE {
    ($vec:expr, $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $MASK:ident) => {
        let permute_mask = _mm256_setr_epi32($a, $b, $c, $d, $e, $f, $g, $h);
        let permuted = _mm256_permutevar8x32_epi32($vec, permute_mask);
        let min = _mm256_min_epi32(permuted, $vec);
        let max = _mm256_max_epi32(permuted, $vec);
        $vec = _mm256_blend_epi32(min, max, $MASK!($a, $b, $c, $d, $e, $f, $g, $h));
    };
}

macro_rules! COEX_SHUFFLE {
    ($vec:expr, $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $MASK:ident) => {{
        const SHUFFLE_MASK: i32 = _MM_SHUFFLE($d, $c, $b, $a);
        let shuffled = _mm256_shuffle_epi32($vec, SHUFFLE_MASK);
        let min = _mm256_min_epi32(shuffled, $vec);
        let max = _mm256_max_epi32(shuffled, $vec);
        $vec = _mm256_blend_epi32(min, max, $MASK!($a, $b, $c, $d, $e, $f, $g, $h));
    }};
}

macro_rules! REVERSE_VEC {
    ($vec:expr) => {
        $vec = _mm256_permutevar8x32_epi32($vec, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
    };
}

macro_rules! SORT_8 {
    ($vec:expr) => {{
        COEX_SHUFFLE!($vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC);
        COEX_SHUFFLE!($vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC);
        COEX_SHUFFLE!($vec, 0, 2, 1, 3, 4, 6, 5, 7, ASC);
        COEX_PERMUTE!($vec, 7, 6, 5, 4, 3, 2, 1, 0, ASC);
        COEX_SHUFFLE!($vec, 2, 3, 0, 1, 6, 7, 4, 5, ASC);
        COEX_SHUFFLE!($vec, 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    }};
}

unsafe fn bitonic_merge_16(vecs: &mut [__m256i], n: usize, s: usize) {
    for t in (s * 2..2 * n).step_by(s * 2) {
        for l in (0..n).step_by(t) {
            for j in (std::cmp::max(l + t - n, 0)..t / 2).step_by(2) {
                REVERSE_VEC!(vecs[l + t - 1 - j]);
                REVERSE_VEC!(vecs[l + t - 2 - j]);
                COEX!(vecs[l + j], vecs[l + t - 1 - j]);
                COEX!(vecs[l + j + 1], vecs[l + t - 2 - j]);
            }
        }
        for m in (t / 2..4).rev().step_by(2) {
            for k in (0..n - m / 2).step_by(m) {
                let bound = std::cmp::min(k + m / 2, n - m / 2);
                for j in (k..bound).step_by(2) {
                    COEX!(vecs[j], vecs[m / 2 + j]);
                    COEX!(vecs[j + 1], vecs[m / 2 + j + 1]);
                }
            }
        }
        for j in (0..n - 2).step_by(4) {
            COEX!(vecs[j], vecs[j + 2]);
            COEX!(vecs[j + 1], vecs[j + 3]);
        }
        for j in (0..n).step_by(2) {
            COEX!(vecs[j], vecs[j + 1]);
        }
        for i in (0..n).step_by(2) {
            COEX_PERMUTE!(vecs[i], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
            COEX_PERMUTE!(vecs[i + 1], 4, 5, 6, 7, 0, 1, 2, 3, ASC);
            let tmp = vecs[i];
            vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
            vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
            COEX!(vecs[i], vecs[i + 1]);
            let tmp = vecs[i];
            vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
            vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
            COEX!(vecs[i], vecs[i + 1]);
            let tmp = vecs[i];
            vecs[i] = _mm256_unpacklo_epi32(vecs[i], vecs[i + 1]);
            vecs[i + 1] = _mm256_unpackhi_epi32(tmp, vecs[i + 1]);
            COEX!(vecs[i], vecs[i + 1]);
        }
    }
}

unsafe fn sort_16_int_vertical(vecs: &mut [__m256i]) {
    COEX!(vecs[0], vecs[1]);
    COEX!(vecs[2], vecs[3]); /* step 1 */
    COEX!(vecs[4], vecs[5]);
    COEX!(vecs[6], vecs[7]);
    COEX!(vecs[8], vecs[9]);
    COEX!(vecs[10], vecs[11]);
    COEX!(vecs[12], vecs[13]);
    COEX!(vecs[14], vecs[15]);
    COEX!(vecs[0], vecs[2]);
    COEX!(vecs[1], vecs[3]); /* step 2 */
    COEX!(vecs[4], vecs[6]);
    COEX!(vecs[5], vecs[7]);
    COEX!(vecs[8], vecs[10]);
    COEX!(vecs[9], vecs[11]);
    COEX!(vecs[12], vecs[14]);
    COEX!(vecs[13], vecs[15]);
    COEX!(vecs[0], vecs[4]);
    COEX!(vecs[1], vecs[5]); /* step 3 */
    COEX!(vecs[2], vecs[6]);
    COEX!(vecs[3], vecs[7]);
    COEX!(vecs[8], vecs[12]);
    COEX!(vecs[9], vecs[13]);
    COEX!(vecs[10], vecs[14]);
    COEX!(vecs[11], vecs[15]);
    COEX!(vecs[0], vecs[8]);
    COEX!(vecs[1], vecs[9]); /* step 4 */
    COEX!(vecs[2], vecs[10]);
    COEX!(vecs[3], vecs[11]);
    COEX!(vecs[4], vecs[12]);
    COEX!(vecs[5], vecs[13]);
    COEX!(vecs[6], vecs[14]);
    COEX!(vecs[7], vecs[15]);
    COEX!(vecs[5], vecs[10]);
    COEX!(vecs[6], vecs[9]); /* step 5 */
    COEX!(vecs[3], vecs[12]);
    COEX!(vecs[7], vecs[11]);
    COEX!(vecs[13], vecs[14]);
    COEX!(vecs[4], vecs[8]);
    COEX!(vecs[1], vecs[2]);
    COEX!(vecs[1], vecs[4]);
    COEX!(vecs[7], vecs[13]); /* step 6 */
    COEX!(vecs[2], vecs[8]);
    COEX!(vecs[11], vecs[14]);
    COEX!(vecs[2], vecs[4]);
    COEX!(vecs[5], vecs[6]); /* step 7 */
    COEX!(vecs[9], vecs[10]);
    COEX!(vecs[11], vecs[13]);
    COEX!(vecs[3], vecs[8]);
    COEX!(vecs[7], vecs[12]);
    COEX!(vecs[3], vecs[5]);
    COEX!(vecs[6], vecs[8]); /* step 8 */
    COEX!(vecs[7], vecs[9]);
    COEX!(vecs[10], vecs[12]);
    COEX!(vecs[3], vecs[4]);
    COEX!(vecs[5], vecs[6]); /* step 9 */
    COEX!(vecs[7], vecs[8]);
    COEX!(vecs[9], vecs[10]);
    COEX!(vecs[11], vecs[12]);
    COEX!(vecs[6], vecs[7]);
    COEX!(vecs[8], vecs[9]); /* step 10 */
}

// auto generated
unsafe fn merge_8_columns_with_16_elements(vecs: &mut [__m256i]) {
    vecs[8] = _mm256_shuffle_epi32(vecs[8], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[7], vecs[8]);
    vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[6], vecs[9]);
    vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[5], vecs[10]);
    vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[4], vecs[11]);
    vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[3], vecs[12]);
    vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[2], vecs[13]);
    vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[1], vecs[14]);
    vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[0], vecs[15]);
    vecs[4] = _mm256_shuffle_epi32(vecs[4], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[3], vecs[4]);
    vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[2], vecs[5]);
    vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[1], vecs[6]);
    vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[0], vecs[7]);
    vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[11], vecs[12]);
    vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[10], vecs[13]);
    vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[9], vecs[14]);
    vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[8], vecs[15]);
    vecs[2] = _mm256_shuffle_epi32(vecs[2], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[1], vecs[2]);
    vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[0], vecs[3]);
    vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[5], vecs[6]);
    vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[4], vecs[7]);
    vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[9], vecs[10]);
    vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[8], vecs[11]);
    vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[13], vecs[14]);
    vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[12], vecs[15]);
    vecs[1] = _mm256_shuffle_epi32(vecs[1], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[0], vecs[1]);
    vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[2], vecs[3]);
    vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[4], vecs[5]);
    vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[6], vecs[7]);
    vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[8], vecs[9]);
    vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[10], vecs[11]);
    vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[12], vecs[13]);
    vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2, 3, 0, 1));
    COEX!(vecs[14], vecs[15]);
    COEX_SHUFFLE!(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    vecs[8] = _mm256_shuffle_epi32(vecs[8], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[7], vecs[8]);
    vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[6], vecs[9]);
    vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[5], vecs[10]);
    vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[4], vecs[11]);
    vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[3], vecs[12]);
    vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[2], vecs[13]);
    vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[1], vecs[14]);
    vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[0], vecs[15]);
    vecs[4] = _mm256_shuffle_epi32(vecs[4], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[3], vecs[4]);
    vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[2], vecs[5]);
    vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[1], vecs[6]);
    vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[0], vecs[7]);
    vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[11], vecs[12]);
    vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[10], vecs[13]);
    vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[9], vecs[14]);
    vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[8], vecs[15]);
    vecs[2] = _mm256_shuffle_epi32(vecs[2], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[1], vecs[2]);
    vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[0], vecs[3]);
    vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[5], vecs[6]);
    vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[4], vecs[7]);
    vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[9], vecs[10]);
    vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[8], vecs[11]);
    vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[13], vecs[14]);
    vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[12], vecs[15]);
    vecs[1] = _mm256_shuffle_epi32(vecs[1], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[0], vecs[1]);
    vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[2], vecs[3]);
    vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[4], vecs[5]);
    vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[6], vecs[7]);
    vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[8], vecs[9]);
    vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[10], vecs[11]);
    vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[12], vecs[13]);
    vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0, 1, 2, 3));
    COEX!(vecs[14], vecs[15]);
    COEX_SHUFFLE!(vecs[0], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[1], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[2], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[3], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[4], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[5], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[6], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[7], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[8], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[9], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[10], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[11], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[12], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[13], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[14], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_SHUFFLE!(vecs[15], 3, 2, 1, 0, 7, 6, 5, 4, ASC);
    COEX_SHUFFLE!(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    REVERSE_VEC!(vecs[8]);
    COEX!(vecs[7], vecs[8]);
    REVERSE_VEC!(vecs[9]);
    COEX!(vecs[6], vecs[9]);
    REVERSE_VEC!(vecs[10]);
    COEX!(vecs[5], vecs[10]);
    REVERSE_VEC!(vecs[11]);
    COEX!(vecs[4], vecs[11]);
    REVERSE_VEC!(vecs[12]);
    COEX!(vecs[3], vecs[12]);
    REVERSE_VEC!(vecs[13]);
    COEX!(vecs[2], vecs[13]);
    REVERSE_VEC!(vecs[14]);
    COEX!(vecs[1], vecs[14]);
    REVERSE_VEC!(vecs[15]);
    COEX!(vecs[0], vecs[15]);
    REVERSE_VEC!(vecs[4]);
    COEX!(vecs[3], vecs[4]);
    REVERSE_VEC!(vecs[5]);
    COEX!(vecs[2], vecs[5]);
    REVERSE_VEC!(vecs[6]);
    COEX!(vecs[1], vecs[6]);
    REVERSE_VEC!(vecs[7]);
    COEX!(vecs[0], vecs[7]);
    REVERSE_VEC!(vecs[12]);
    COEX!(vecs[11], vecs[12]);
    REVERSE_VEC!(vecs[13]);
    COEX!(vecs[10], vecs[13]);
    REVERSE_VEC!(vecs[14]);
    COEX!(vecs[9], vecs[14]);
    REVERSE_VEC!(vecs[15]);
    COEX!(vecs[8], vecs[15]);
    REVERSE_VEC!(vecs[2]);
    COEX!(vecs[1], vecs[2]);
    REVERSE_VEC!(vecs[3]);
    COEX!(vecs[0], vecs[3]);
    REVERSE_VEC!(vecs[6]);
    COEX!(vecs[5], vecs[6]);
    REVERSE_VEC!(vecs[7]);
    COEX!(vecs[4], vecs[7]);
    REVERSE_VEC!(vecs[10]);
    COEX!(vecs[9], vecs[10]);
    REVERSE_VEC!(vecs[11]);
    COEX!(vecs[8], vecs[11]);
    REVERSE_VEC!(vecs[14]);
    COEX!(vecs[13], vecs[14]);
    REVERSE_VEC!(vecs[15]);
    COEX!(vecs[12], vecs[15]);
    REVERSE_VEC!(vecs[1]);
    COEX!(vecs[0], vecs[1]);
    REVERSE_VEC!(vecs[3]);
    COEX!(vecs[2], vecs[3]);
    REVERSE_VEC!(vecs[5]);
    COEX!(vecs[4], vecs[5]);
    REVERSE_VEC!(vecs[7]);
    COEX!(vecs[6], vecs[7]);
    REVERSE_VEC!(vecs[9]);
    COEX!(vecs[8], vecs[9]);
    REVERSE_VEC!(vecs[11]);
    COEX!(vecs[10], vecs[11]);
    REVERSE_VEC!(vecs[13]);
    COEX!(vecs[12], vecs[13]);
    REVERSE_VEC!(vecs[15]);
    COEX!(vecs[14], vecs[15]);
    COEX_PERMUTE!(vecs[0], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[0], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[1], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[1], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[2], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[2], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[3], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[3], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[4], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[4], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[5], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[5], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[6], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[6], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[7], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[7], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[8], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[8], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[9], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[9], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[10], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[10], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[11], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[11], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[12], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[12], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[13], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[13], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[14], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[14], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
    COEX_PERMUTE!(vecs[15], 7, 6, 5, 4, 3, 2, 1, 0, ASC);
    COEX_SHUFFLE!(vecs[15], 2, 3, 0, 1, 6, 7, 4, 5, ASC);
    COEX_SHUFFLE!(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC);
}

#[cfg(test)]
mod test {
    use core::arch::x86_64::_mm256_set_epi32;
    use std::mem::transmute;

    use super::*;

    #[test]
    fn test_sort_16() {
        unsafe {
            let mut v1 = _mm256_set_epi32(16, 14, 12, 10, 8, 6, 4, 2);
            let mut v2 = _mm256_set_epi32(15, 13, 11, 9, 7, 5, 3, 1);
            let v1_arr: [i32; 8] = transmute(v1);
            let v2_arr: [i32; 8] = transmute(v2);
            println!("{:?} {:?}", v1_arr, v2_arr);
            SORT_16!(v1, v2);
            let v1_arr: [i32; 8] = transmute(v1);
            let v2_arr: [i32; 8] = transmute(v2);
            println!("{:?} {:?}", v1_arr, v2_arr);
        }
    }

    #[test]
    fn test_sort_8() {
        unsafe {
            let mut v1 = _mm256_set_epi32(16, 14, 12, 10, 8, 6, 4, 2);
            let v1_arr: [i32; 8] = transmute(v1);
            println!("{:?}", v1_arr);
            SORT_8!(v1);
            let v1_arr: [i32; 8] = transmute(v1);
            println!("{:?}", v1_arr);
        }
    }
}
