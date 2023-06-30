# Rust Simd Sort

Simd sorting implementation based on [intel simd sort](https://github.com/intel/x86-simd-sort/tree/main) with support for various architectures

## Supported DataTypes

|   | i64 |
|---|-----|
| avx2 | ✓ |
| avx512 | ✓ |
| wasm-simd128 | ✓ |
| portable-simd | ✓ |