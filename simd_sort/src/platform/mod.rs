#[cfg(feature = "nightly")]
pub mod nightly;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_family = "wasm")]
pub mod wasm;

pub fn sort_i64(data: &mut [i64]) {
    #[cfg(all(target_arch = "x86_64", not(feature = "nightly")))]
    {
        if cfg!(feature = "avx2") {
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
        if cfg!(feature = "avx2") {
            x86::avx2::avx2_sort_i64(data)
        } else {
            data.sort_unstable()
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    data.sort_unstable()
}
