use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::{distributions::Uniform, Rng};
use simd_sort::{
    self, platform::nightly::portable_simd_sort_i64, platform::x86::avx2::avx2_sort_i64,
};

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
    let data2 = data.clone();
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

    c.bench_function("portable_simd_sort", move |b| {
        // This will avoid timing the to_vec call.
        b.iter_batched(
            || data2.clone(),
            |mut data| {
                portable_simd_sort_i64(data.as_mut_slice());
                black_box(data);
            },
            BatchSize::LargeInput,
        )
    });

    c.bench_function("avx2", move |b| {
        // This will avoid timing the to_vec call.
        b.iter_batched(
            || data.clone(),
            |mut data| {
                avx2_sort_i64(data.as_mut_slice());
                black_box(data);
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
