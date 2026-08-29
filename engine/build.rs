use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=assets/tinyccrl.nnue");
    println!("cargo:rustc-check-cfg=cfg(nnue_asset)");
    if Path::new("assets/tinyccrl.nnue").exists() {
        println!("cargo:rustc-cfg=nnue_asset");
    }
}
