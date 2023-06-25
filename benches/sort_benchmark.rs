#![cfg_attr(feature = "nightly", feature(portable_simd))]

use std::fmt::Debug;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::{distributions::Uniform, Rng};
use simd_sort;

pub fn create_uniform_data<T>(size: usize) -> Vec<T>
where
    T: TryFrom<usize> + Default,
    <T as TryFrom<usize>>::Error: Debug,
{
    let std_rng = rand::SeedableRng::seed_from_u64(10);
    let mut rng = rand::rngs::StdRng::from(std_rng);
    let range = Uniform::new(0, size);
    let vals: Vec<T> = (0..size)
        .map(|_| rng.sample(&range).try_into().unwrap_or_default())
        .collect();
    vals
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let data = create_uniform_data::<i64>(1024 * 1024);

    #[cfg(feature = "rust_std_unstable")]
    {
        let data_t = data.clone();
        c.bench_function("std_unstable_sort", move |b| {
            // This will avoid timing the to_vec call.
            b.iter_batched(
                || data_t.clone(),
                |mut data| {
                    data.sort_unstable();
                    black_box(data);
                },
                BatchSize::LargeInput,
            )
        });
    }

    #[cfg(any(feature = "cpp_vqsort", feature = "cpp_vqsort_avx2"))]
    {
        extern "C" {
            fn vqsort_i64(data: *mut i64, size: usize);
        }
        unsafe {
            let mut temp1 = data.clone();
            let mut temp2 = data.clone();
            vqsort_i64(temp1.as_mut_ptr(), temp1.len());
            temp2.sort();
            assert_eq!(temp1, temp2);
            let data_t = data.clone();
            c.bench_function("cpp_vqsort", move |b| {
                // This will avoid timing the to_vec call.
                b.iter_batched(
                    || data_t.clone(),
                    |mut data| {
                        vqsort_i64(data.as_mut_ptr(), data.len());
                        black_box(data);
                    },
                    BatchSize::LargeInput,
                )
            });
        }
    }

    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        use simd_sort::platform::x86::avx2::avx2_sort_i64;
        let data_t = data.clone();
        c.bench_function("avx2", move |b| {
            // This will avoid timing the to_vec call.
            b.iter_batched(
                || data_t.clone(),
                |mut data| {
                    avx2_sort_i64(data.as_mut_slice());
                    black_box(data);
                },
                BatchSize::LargeInput,
            )
        });
    }

    #[cfg(all(target_feature = "avx512f", target_arch = "x86_64"))]
    {
        use simd_sort::platform::x86::avx512::avx512_sort_i64;
        let data_t = data.clone();
        c.bench_function("avx512", move |b| {
            // This will avoid timing the to_vec call.
            b.iter_batched(
                || data_t.clone(),
                |mut data| {
                    avx512_sort_i64(data.as_mut_slice());
                    black_box(data);
                },
                BatchSize::LargeInput,
            )
        });
    }

    #[cfg(all(target_feature = "simd128", target_arch = "wasm32"))]
    {
        use simd_sort::platform::wasm::wasm128_sort_i64;
        let data_t = data.clone();
        c.bench_function("wasm32", move |b| {
            // This will avoid timing the to_vec call.
            b.iter_batched(
                || data_t.clone(),
                |mut data| {
                    wasm128_sort_i64(data.as_mut_slice());
                    black_box(data);
                },
                BatchSize::LargeInput,
            )
        });
    }


    #[cfg(feature = "nightly")]
    {
        use simd_sort::platform::nightly::portable_simd_sort_i64;
        let data_t = data.clone();
        c.bench_function("portable_simd_sort", move |b| {
            // This will avoid timing the to_vec call.
            b.iter_batched(
                || data_t.clone(),
                |mut data| {
                    portable_simd_sort_i64(data.as_mut_slice());
                    black_box(data);
                },
                BatchSize::LargeInput,
            )
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
