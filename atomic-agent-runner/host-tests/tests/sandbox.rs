//! The test host's WASI context is a sandbox: a fixture guest that reads its
//! environment, arguments and working directory sees nothing.
//!
//! This is its own test binary with a single test because it sets a process
//! environment variable, which is only sound while no other thread reads
//! the environment.

use atomic_agent_runner_host_tests::probe_sandbox;
use serde_json::json;

#[test]
fn sandbox_gives_no_env_args_or_filesystem() {
    // SAFETY: the only test in this binary, set before any thread is spawned
    // by it and before anything reads the environment.
    unsafe { std::env::set_var("ATOMIC_AGENT_RUNNER_PROBE", "leak") };
    assert_eq!(
        std::env::var("ATOMIC_AGENT_RUNNER_PROBE").as_deref(),
        Ok("leak")
    );

    let seen = probe_sandbox();
    // Names only: on a leak the values are the test process's own
    // environment, which may hold secrets that must not reach test logs.
    let leaked: Vec<&str> = seen["vars"]
        .as_array()
        .expect("vars is an array")
        .iter()
        .map(|kv| kv[0].as_str().unwrap_or("?"))
        .collect();
    assert!(
        leaked.is_empty(),
        "guest saw environment variables {leaked:?}"
    );
    assert_eq!(seen["args"], json!([]), "guest saw arguments");
    assert_eq!(
        seen["cwd_readable"],
        json!(false),
        "guest could read its cwd"
    );
}
