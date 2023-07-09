#![cfg_attr(feature = "nightly", feature(portable_simd))]

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use paste::paste;
use rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};
use simd_sort;

pub fn create_uniform_data<T>(size: usize) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(42);
    let vals: Vec<T> = (0..size).map(|_| rng.gen::<T>()).collect();
    vals
}

fn slice_sort_unstable<T>(data: &mut [T])
where
    T: Ord,
{
    data.sort_unstable();
}

fn slice_sort_unstable_by<T>(data: &mut [T])
where
    T: PartialOrd,
{
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
}

macro_rules! rust_std_unstable {
    ($c: ident, $ty: ident, $name: literal, $sort_fn: ident) => {
        #[cfg(feature = "rust_std_unstable")]
        {
            let data_t = create_uniform_data::<$ty>(1024 * 1024);
            $c.bench_function($name, move |b| {
                // This will avoid timing the to_vec call.
                b.iter_batched(
                    || data_t.clone(),
                    |mut data| {
                        $sort_fn(&mut data);
                        black_box(data);
                    },
                    BatchSize::LargeInput,
                )
            });
        }
    };
}

macro_rules! cpp_vqsort {
    ($c: ident, $ty: ident, $name: literal, $sort_fn: ident) => {
        paste! {
            #[cfg(feature = "cpp_vqsort")]
            {
                let data = create_uniform_data::<$ty>(1024 * 1024);
                extern "C" {
                    fn [<vqsort_ $ty>](data: *mut $ty, size: usize);
                }
                unsafe {
                    let mut temp1 = data.clone();
                    let mut temp2 = data.clone();
                    [<vqsort_ $ty>](temp1.as_mut_ptr(), temp1.len());
                    $sort_fn(&mut temp2);
                    assert_eq!(temp1, temp2);
                    let data_t = data.clone();
                    $c.bench_function($name, move |b| {
                        // This will avoid timing the to_vec call.
                        b.iter_batched(
                            || data_t.clone(),
                            |mut data| {
                                [<vqsort_ $ty>](data.as_mut_ptr(), data.len());
                                black_box(data);
                            },
                            BatchSize::LargeInput,
                        )
                    });
                }
            }
        }
    };
}

macro_rules! avx2_sort {
    ($c: ident, $ty: ident, $name: literal, $sort_fn: ident) => {
        paste! {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") {
                    let data = create_uniform_data::<$ty>(1024 * 1024);
                    use simd_sort::platform::x86::avx2::$sort_fn;
                    let data_t = data.clone();
                    $c.bench_function($name, move |b| {
                        // This will avoid timing the to_vec call.
                        b.iter_batched(
                            || data_t.clone(),
                            |mut data| {
                                $sort_fn(data.as_mut_slice());
                                black_box(data);
                            },
                            BatchSize::LargeInput,
                        )
                    });
                }
            }
        }
    };
}

macro_rules! cpp_avx512_qsort {
    ($c: ident, $ty: ident, $name: literal, $sort_fn: ident) => {
        paste! {
            #[cfg(all(feature = "cpp_avx512_qsort", target_arch = "x86_64"))]
            {
                if std::is_x86_feature_detected!("avx512f") {
                    let data = create_uniform_data::<$ty>(1024 * 1024);
                    extern "C" {
                        fn [<avx512_qsort_ $ty>](data: *mut $ty, size: usize);
                    }
                    unsafe {
                        let mut temp1 = data.clone();
                        let mut temp2 = data.clone();
                        [<avx512_qsort_ $ty>](temp1.as_mut_ptr(), temp1.len());
                        $sort_fn(&mut temp2);
                        assert_eq!(temp1, temp2);
                        let data_t = data.clone();
                        $c.bench_function($name, move |b| {
                            // This will avoid timing the to_vec call.
                            b.iter_batched(
                                || data_t.clone(),
                                |mut data| {
                                    [<avx512_qsort_ $ty>](data.as_mut_ptr(), data.len());
                                    black_box(data);
                                },
                                BatchSize::LargeInput,
                            )
                        });
                    }
                }
            }
        }
    };
}

macro_rules! avx512_sort {
    ($c: ident, $ty: ident, $name: literal, $sort_fn: ident) => {
        paste! {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx512f") {
                    let data = create_uniform_data::<$ty>(1024 * 1024);
                    use simd_sort::platform::x86::avx512::$sort_fn;
                    let data_t = data.clone();
                    $c.bench_function($name, move |b| {
                        // This will avoid timing the to_vec call.
                        b.iter_batched(
                            || data_t.clone(),
                            |mut data| {
                                $sort_fn(data.as_mut_slice());
                                black_box(data);
                            },
                            BatchSize::LargeInput,
                        )
                    });
                }
            }
        }
    };
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort benchmarks");
    group.throughput(Throughput::Bytes(1024 * 1024 * 8));

    rust_std_unstable!(group, i64, "rust_std_unstable_i64", slice_sort_unstable);
    rust_std_unstable!(group, u64, "rust_std_unstable_u64", slice_sort_unstable);
    rust_std_unstable!(group, f64, "rust_std_unstable_f64", slice_sort_unstable_by);

    cpp_vqsort!(group, i64, "cpp_vqsort_i64", slice_sort_unstable);
    cpp_vqsort!(group, u64, "cpp_vqsort_u64", slice_sort_unstable);
    cpp_vqsort!(group, f64, "cpp_vqsort_f64", slice_sort_unstable_by);

    avx2_sort!(group, i64, "avx2_i64", avx2_sort_i64);
    avx2_sort!(group, f64, "avx2_f64", avx2_sort_f64);

    cpp_avx512_qsort!(group, i64, "cpp_avx512_qsort_i64", slice_sort_unstable);
    cpp_avx512_qsort!(group, u64, "cpp_avx512_qsort_u64", slice_sort_unstable);
    cpp_avx512_qsort!(group, f64, "cpp_avx512_qsort_f64", slice_sort_unstable_by);

    avx512_sort!(group, i64, "avx512_i64", avx512_sort_i64);
    avx512_sort!(group, f64, "avx512_f64", avx512_sort_f64);
    avx512_sort!(group, u64, "avx512_u64", avx512_sort_u64);

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

    let data = create_uniform_data::<i64>(1024 * 1024);
    {
        use simd_sort::platform::sort_i64;
        let data_t = data.clone();
        group.bench_function("best_runtime", move |b| {
            // This will avoid timing the to_vec call.
            b.iter_batched(
                || data_t.clone(),
                |mut data| {
                    sort_i64(data.as_mut_slice());
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
        group.bench_function("portable_simd_sort", move |b| {
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
