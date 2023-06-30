#[cfg(target_feature = "avx2")]
pub mod avx2;

#[cfg(feature = "nightly")]
pub mod avx512;
