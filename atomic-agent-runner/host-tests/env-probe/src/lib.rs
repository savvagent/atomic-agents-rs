//! Fixture guest for the host tests' sandbox check.

wit_bindgen::generate!({ world: "env-probe", path: "wit" });

struct Probe;

impl Guest for Probe {
    fn probe() -> String {
        let vars: Vec<[String; 2]> = std::env::vars().map(|(k, v)| [k, v]).collect();
        let args: Vec<String> = std::env::args().collect();
        serde_json::json!({
            "vars": vars,
            "args": args,
            "cwd_readable": std::fs::read_dir(".").is_ok(),
        })
        .to_string()
    }
}

export!(Probe);
