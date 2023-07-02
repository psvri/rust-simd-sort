#[cfg(feature = "nightly")]
pub mod nightly;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_family = "wasm")]
pub mod wasm;

pub fn sort_i64(data: &mut [i64]) {
    #[cfg(all(target_arch = "x86_64", not(feature = "nightly")))]
    {
        if cfg!(target_feature = "avx2") {
            x86::avx2::avx2_sort_i64(data)
        } else {
            data.sort_unstable()
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        if cfg!(target_feature = "avx512f") {
            x86::avx512::avx512_sort_i64(data)
        }
        if cfg!(target_feature = "avx2") {
            x86::avx2::avx2_sort_i64(data)
        } else {
            data.sort_unstable()
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    data.sort_unstable()
}

fn compress_store_naive(input: &[i64], output: &mut [i64], mask: u8) {
    unsafe {
        let mut temp_mask = mask;
        let mut base_ptr = output.as_mut_ptr();
        let value_ptr: *const i64 = std::mem::transmute(&input);

        for i in 0..8 {
            if temp_mask & 1 == 1 {
                *base_ptr = *value_ptr.offset(i);
                base_ptr = base_ptr.offset(1);
            }
            temp_mask = temp_mask >> 1;
        }
    }
}

pub fn compress_store_i64(input: &[i64], output: &mut [i64], mask: u8) {
    #[cfg(all(target_arch = "x86_64", not(feature = "nightly")))]
    {
        if cfg!(target_feature = "avx2") {
            x86::avx2::avx2_compress_store_i64(input, output, mask);
        } else {
            compress_store_naive(input, output, mask);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        if cfg!(target_feature = "avx2") {
            x86::avx2::avx2_compress_store_i64(input, output, mask);
        } else {
            compress_store_naive(input, output, mask);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    compress_store_naive(input, output, mask);
}
