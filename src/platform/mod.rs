#[cfg(feature = "nightly")]
pub mod nightly;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_family = "wasm")]
pub mod wasm;
