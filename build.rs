use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=c_src/ring_crc.h");
    println!("cargo:rerun-if-changed=c_src/ring_crc.c");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let obj_path = out_dir.join("ring_crc.o");
    let lib_path = out_dir.join("libring_crc.a");

    let cc = env::var("CC").unwrap_or_else(|_| "gcc".to_string());
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());

    let status = Command::new(&cc)
        .args([
            "-std=c11",
            "-O3",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wpedantic",
            "-fPIC",
            "-Ic_src",
            "-c",
            "c_src/ring_crc.c",
            "-o",
        ])
        .arg(&obj_path)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke C11 compiler `{cc}`: {e}"));

    assert!(status.success(), "C11 compilation of c_src/ring_crc.c failed");

    let ar_status = Command::new(&ar)
        .arg("rcs")
        .arg(&lib_path)
        .arg(&obj_path)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke archiver `{ar}`: {e}"));

    assert!(ar_status.success(), "static library archive creation failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ring_crc");
}
