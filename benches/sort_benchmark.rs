#![cfg_attr(feature = "nightly", feature(portable_simd))]

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::{distributions::Uniform, Rng};
use simd_sort::{self, };

pub fn create_uniform_data() -> Vec<u64> {
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0, 1024 * 1024);
    let vals: Vec<u64> = (0..1024 * 1024).map(|_| rng.sample(&range)).collect();
    vals
}

pub fn create_uniform_data_i64() -> Vec<i64> {
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0, 1024 * 1024);
    let vals: Vec<i64> = (0..1024 * 1024).map(|_| rng.sample(&range)).collect();
    vals
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let data = create_uniform_data_i64();
    let data1 = data.clone();

    c.bench_function("std_unstable_sort", move |b| {
        // This will avoid timing the to_vec call.
        b.iter_batched(
            || data1.clone(),
            |mut data| {
                data.sort_unstable();
                black_box(data);
            },
            BatchSize::LargeInput,
        )
    });

    #[cfg(target_feature = "avx2")]
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
    
    #[cfg(target_feature = "avx512f")]
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
