pub mod common;
pub mod bit_64;

pub trait PortableSimdSort {
    fn sort_portable_simd(&mut self);
}