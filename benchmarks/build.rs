#![allow(dead_code)]

use std::{env, path::PathBuf};

//choose which ever you like
//static CLANG_PATH: &'static str = "clang++-12";
static CLANG_PATH: &'static str = "g++-10";

fn get_manifest_dir_path() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
}

fn build_and_link_cpp_sort(
    file_name: &str,
    specialize_fn: Option<fn(&mut cc::Build) -> Option<String>>,
) {
    let file_path = get_manifest_dir_path()
        .join("other_implementations/")
        .join(format!("{file_name}.cpp"));

    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed={}", file_path.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let mut builder = cc::Build::new();

    builder
        .file(file_path)
        .cpp(true)
        .warnings(false) // The thirdparties just have too many.
        .flag_if_supported("/EHsc")
        .flag_if_supported("/Zc:__cplusplus")
        .flag_if_supported("/std:c++20")
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-fdiagnostics-color=always")
        .force_frame_pointer(false)
        .define("NDEBUG", None)
        .define("HWY_LIBRARY_TYPE", "STATIC")
        .debug(false)
        .opt_level(3);

    let mut artifact_name = file_name.to_string();
    if let Some(spec_fn) = specialize_fn {
        if let Some(artifact_name_override) = spec_fn(&mut builder) {
            artifact_name = artifact_name_override;
        }
    }
    builder.compile(&artifact_name);

    println!("{:?} {:?}", out_dir.display(), artifact_name);
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={}", artifact_name);
}

#[cfg(all(feature = "cpp_vqsort", target_arch = "aarch64"))]
fn build_and_link_cpp_vqsort() {
    build_and_link_cpp_sort(
        "cpp_vqsort",
        Some(|builder: &mut cc::Build| {
            builder.flag("-march=native+aes");
            builder.include(
                get_manifest_dir_path()
                    .join("other_implementations")
                    .join("vqsort"),
            );
            builder.compiler(CLANG_PATH);

            None
        }),
    );
}

#[cfg(all(feature = "cpp_vqsort", target_arch = "x86_64"))]
fn build_and_link_cpp_vqsort() {
    build_and_link_cpp_sort(
        "cpp_vqsort",
        Some(|builder: &mut cc::Build| {
            builder.flag("-march=native");
            builder.include(
                get_manifest_dir_path()
                    .join("other_implementations")
                    .join("vqsort"),
            );
            builder.compiler(CLANG_PATH);

            None
        }),
    );
}

#[cfg(all(feature = "cpp_vqsort_avx2", target_arch = "x86_64"))]
fn build_and_link_cpp_vqsort_avx2() {
    build_and_link_cpp_sort(
        "cpp_vqsort",
        Some(|builder: &mut cc::Build| {
            builder.flag("-march=x86-64-v3");
            builder.include(get_manifest_dir_path().join("other_implementations/vqsort"));
            builder.compiler(CLANG_PATH);
            None
        }),
    );
}

#[cfg(all(feature = "cpp_avx512_qsort", target_arch = "x86_64"))]
fn build_and_link_cpp_avx512_qsort() {
    build_and_link_cpp_sort(
        "cpp_avx512_qsort",
        Some(|builder: &mut cc::Build| {
            builder.flag("-march=native");
            builder
                .include(get_manifest_dir_path().join("other_implementations/x86-simd-sort/src"));
            builder.compiler(CLANG_PATH);
            None
        }),
    );
}

#[cfg(not(feature = "cpp_vqsort"))]
fn build_and_link_cpp_vqsort() {}

#[cfg(not(feature = "cpp_vqsort_avx2"))]
fn build_and_link_cpp_vqsort_avx2() {}

#[cfg(not(feature = "cpp_avx512_qsort"))]
fn build_and_link_cpp_avx512_qsort() {}

fn main() {
    build_and_link_cpp_vqsort();
    build_and_link_cpp_vqsort_avx2();
    build_and_link_cpp_avx512_qsort();
}
