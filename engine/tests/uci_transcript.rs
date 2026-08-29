use std::process::{Command, Stdio};

fn transcript(script: &str) -> String {
    let fixture = format!(
        "{}/assets/fixtures/tinyccrl-test.nnue",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut child = Command::new(env!("CARGO_BIN_EXE_tinyccrl-engine"))
        .env("NNUE_WEIGHTS", fixture)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start engine");
    {
        use std::io::Write;
        child
            .stdin
            .take()
            .expect("engine stdin")
            .write_all(script.as_bytes())
            .expect("write UCI script");
    }
    let output = child.wait_with_output().expect("read engine output");
    assert!(output.status.success());
    String::from_utf8(output.stdout).expect("UTF-8 engine output")
}

#[test]
fn startpos_nodes_transcript() {
    assert_eq!(
        transcript("position startpos\ngo nodes 100\nquit\n"),
        "info nodes 100\nbestmove a2a4\n"
    );
}

#[test]
fn kiwipete_nodes_transcript() {
    assert_eq!(
        transcript(
            "position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\ngo nodes 100\nquit\n"
        ),
        "info nodes 100\nbestmove a2a3\n"
    );
}

#[test]
fn endgame_nodes_transcript() {
    assert_eq!(
        transcript("position fen 8/8/3k4/8/3K4/8/8/8 b - - 0 1\ngo nodes 25\nquit\n"),
        "info nodes 25\nbestmove d6c6\n"
    );
}
