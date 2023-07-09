use std::arch::x86_64::{
    __m512i, __mmask8, _mm512_castpd_si512, _mm512_castsi512_pd, _mm512_permutexvar_epi64,
    _mm512_set_epi64, _mm512_shuffle_pd, _MM_PERM_ENUM,
};

pub const SHUFFLE1_1_1_1: _MM_PERM_ENUM = shuffle_mask([1, 1, 1, 1]);
pub const SHUFFLE2_0XAA_MASK: __mmask8 = 0xAA;
pub const SHUFFLE2_0XCC_MASK: __mmask8 = 0xCC;
pub const SHUFFLE2_0XF0_MASK: __mmask8 = 0xF0;

const fn shuffle_mask(a: [_MM_PERM_ENUM; 4]) -> _MM_PERM_ENUM {
    (a[0] << 6) | (a[1] << 4) | (a[2] << 2) | a[3]
}

pub fn shuffle_m512<const MASK: _MM_PERM_ENUM>(zmm: __m512i) -> __m512i {
    unsafe {
        let temp = _mm512_castsi512_pd(zmm);
        _mm512_castpd_si512(_mm512_shuffle_pd::<MASK>(temp, temp))
    }
}

pub fn permutexvar_m512(idx: __m512i, a: __m512i) -> __m512i {
    unsafe { _mm512_permutexvar_epi64(idx, a) }
}

pub fn network64bit1_idx() -> __m512i {
    unsafe { _mm512_set_epi64(4, 5, 6, 7, 0, 1, 2, 3) }
}

pub fn network64bit2_idx() -> __m512i {
    unsafe { _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7) }
}

pub fn network64bit3_idx() -> __m512i {
    unsafe { _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2) }
}

pub fn network64bit4_idx() -> __m512i {
    unsafe { _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4) }
}

#[cfg(test)]
#[cfg(target_feature = "avx512f")]
pub mod test {
    use std::ops::*;

    pub fn generate_mask_answer<T, M>(bitmask: M, values: &[T]) -> (M, [T; 8])
    where
        T: Default + Copy,
        M: BitAnd<u8> + std::marker::Copy,
        <M as BitAnd<u8>>::Output: PartialEq<u8>,
    {
        let mut new_values = [<T as Default>::default(); 8];
        let mut count = 0;
        for i in 0..8 {
            if bitmask & (1 << i) != 0 {
                new_values[count] = values[i];
                count += 1;
            }
        }
        (bitmask, new_values)
    }

    pub fn mask_fn(x: u8) -> u8 {
        x
    }
}
