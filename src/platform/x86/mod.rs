#[cfg(target_feature = "avx2")]
pub mod avx2;

#[cfg(all(target_feature = "avx512f", feature = "nightly"))]
pub mod avx512;
