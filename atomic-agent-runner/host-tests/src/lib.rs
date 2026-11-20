//! A wasmtime host for the `atomic-agent-runner` component, for tests.
//!
//! [`run`] instantiates the built component with a scripted model
//! ([`Script`]), recorded tool calls, an in-memory state store, captured logs
//! and a fixed clock, then calls its `run` export. [`runner_imports`] lists
//! what the component imports, so a test can hold it to the world's
//! interfaces plus the std-runtime allowlist.
//!
//! The component comes from [`component_path`]: the file named by
//! `ATOMIC_AGENT_RUNNER_WASM` when that is set (how a release asset is
//! checked), otherwise a fresh `cargo build` for `wasm32-wasip2`.
//!
//! Every instantiation gets its WASI context from [`sandbox_wasi_ctx`], the
//! sandbox a host is expected to give the guest.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde_json::Value;
use wasmtime::component::{Component, HasSelf, Linker, ResourceTable};
use wasmtime::{Engine, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

mod bindings {
    wasmtime::component::bindgen!({
        world: "runner",
        path: "../wit",
        additional_derives: [PartialEq],
    });

    pub mod probe {
        wasmtime::component::bindgen!({
            world: "env-probe",
            path: "env-probe/wit",
        });
    }
}

use bindings::Runner;
pub use bindings::atomic_agents::runner::types::{
    CallOrigin, Completion, CompletionRequest, ContentBlock, Level, LlmError, Message, Role,
    RunError, StepFailure, ToolCall, ToolError, ToolInfo, ToolResult, ToolSpec, Usage,
};
use bindings::atomic_agents::runner::{clock, llm, log, mcp, state, types};
use bindings::probe::EnvProbe;

/// Environment variable naming a prebuilt component to test instead of
/// building one.
pub const WASM_ENV: &str = "ATOMIC_AGENT_RUNNER_WASM";

/// What the test host's `clock.now` always returns, in ms since the epoch.
pub const FIXED_NOW_MS: u64 = 1_700_000_000_000;

/// The workspace root (two levels above this crate).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root exists")
}

/// Builds a `wasm32-wasip2` release artifact into `<workspace>/target/wasm-component`
/// and returns the path of `file` inside it. The separate target directory
/// keeps this nested build off the lock held by the `cargo test` running us.
fn build_wasm(args: &[&str], file: &str) -> PathBuf {
    let root = workspace_root();
    let target_dir = root.join("target/wasm-component");
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let out = Command::new(cargo)
        .current_dir(&root)
        .arg("build")
        .args(args)
        .args(["--release", "--target", "wasm32-wasip2", "--target-dir"])
        .arg(&target_dir)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "could not run {:?}: {e}",
                std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into())
            )
        });
    assert!(
        out.status.success(),
        "building {file} for wasm32-wasip2 failed. If the target is missing, install \
         it with `rustup target add wasm32-wasip2` (for the runner, setting {WASM_ENV} \
         to a built component skips this build).\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let path = target_dir.join("wasm32-wasip2/release").join(file);
    assert!(path.is_file(), "{} was not produced", path.display());
    path
}

/// The runner component under test: `$ATOMIC_AGENT_RUNNER_WASM` if set,
/// otherwise built once per test process.
pub fn component_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| match std::env::var_os(WASM_ENV) {
        Some(p) => {
            let p = PathBuf::from(p);
            assert!(p.is_file(), "{WASM_ENV}={} is not a file", p.display());
            p
        }
        None => build_wasm(&["-p", "atomic-agent-runner"], "atomic_agent_runner.wasm"),
    })
    .clone()
}

/// The `env-probe` fixture guest, built once per test process.
pub fn env_probe_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("env-probe/Cargo.toml");
        let manifest = manifest.to_str().expect("UTF-8 path");
        build_wasm(&["--manifest-path", manifest], "env_probe.wasm")
    })
    .clone()
}

fn engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(Engine::default)
}

fn runner_component() -> &'static Component {
    static COMPONENT: OnceLock<Component> = OnceLock::new();
    COMPONENT.get_or_init(|| {
        Component::from_file(engine(), component_path()).expect("runner component loads")
    })
}

/// The import names of the runner component, in declaration order.
pub fn runner_imports() -> Vec<String> {
    runner_component()
        .component_type()
        .imports(engine())
        .map(|(name, _)| name.to_string())
        .collect()
}

/// The WASI context every guest in these tests runs with: empty environment,
/// no arguments, no preopened directories, stdin/stdout/stderr not inherited
/// (closed stdin, discarded output) and no network (TCP, UDP and name lookup
/// off; the default socket check already denies every address).
pub fn sandbox_wasi_ctx() -> WasiCtx {
    WasiCtx::builder()
        .allow_tcp(false)
        .allow_udp(false)
        .allow_ip_name_lookup(false)
        .build()
}

/// Links the std-runtime WASI interfaces the runner may import, and nothing
/// else: no `wasi:filesystem` and no `wasi:sockets`, so a component that
/// needs them fails to instantiate here as well as failing the import test.
fn add_runner_wasi(l: &mut Linker<TestHost>) -> wasmtime::Result<()> {
    use wasmtime_wasi::cli::{WasiCli, WasiCliView};
    use wasmtime_wasi::clocks::{WasiClocks, WasiClocksView};
    use wasmtime_wasi::p2::bindings::{cli, clocks, random, sync};
    use wasmtime_wasi::random::WasiRandom;

    struct Io;
    impl wasmtime::component::HasData for Io {
        type Data<'a> = &'a mut ResourceTable;
    }

    sync::io::error::add_to_linker::<TestHost, Io>(l, |t| &mut t.table)?;
    sync::io::poll::add_to_linker::<TestHost, Io>(l, |t| &mut t.table)?;
    sync::io::streams::add_to_linker::<TestHost, Io>(l, |t| &mut t.table)?;
    clocks::wall_clock::add_to_linker::<TestHost, WasiClocks>(l, TestHost::clocks)?;
    clocks::monotonic_clock::add_to_linker::<TestHost, WasiClocks>(l, TestHost::clocks)?;
    random::random::add_to_linker::<TestHost, WasiRandom>(l, |t| t.wasi.random())?;
    random::insecure::add_to_linker::<TestHost, WasiRandom>(l, |t| t.wasi.random())?;
    random::insecure_seed::add_to_linker::<TestHost, WasiRandom>(l, |t| t.wasi.random())?;
    cli::exit::add_to_linker::<TestHost, WasiCli>(l, TestHost::cli)?;
    cli::environment::add_to_linker::<TestHost, WasiCli>(l, TestHost::cli)?;
    cli::stdin::add_to_linker::<TestHost, WasiCli>(l, TestHost::cli)?;
    cli::stdout::add_to_linker::<TestHost, WasiCli>(l, TestHost::cli)?;
    cli::stderr::add_to_linker::<TestHost, WasiCli>(l, TestHost::cli)?;
    cli::terminal_input::add_to_linker::<TestHost, WasiCli>(l, TestHost::cli)?;
    cli::terminal_output::add_to_linker::<TestHost, WasiCli>(l, TestHost::cli)?;
    cli::terminal_stdin::add_to_linker::<TestHost, WasiCli>(l, TestHost::cli)?;
    cli::terminal_stdout::add_to_linker::<TestHost, WasiCli>(l, TestHost::cli)?;
    cli::terminal_stderr::add_to_linker::<TestHost, WasiCli>(l, TestHost::cli)?;
    Ok(())
}

/// What the scripted host answers: model completions in order, the tools it
/// lists, per-tool results in order, and the state before the run.
#[derive(Default)]
pub struct Script {
    completions: VecDeque<Result<Completion, LlmError>>,
    tools: Vec<ToolInfo>,
    results: HashMap<String, VecDeque<Result<String, ToolError>>>,
    state: HashMap<String, String>,
}

impl Script {
    /// A host that lists `tools`. Other names can still be called directly,
    /// as act tools are.
    pub fn new(tools: &[&str]) -> Self {
        Self {
            tools: tools
                .iter()
                .map(|n| ToolInfo {
                    name: (*n).to_string(),
                    description: format!("{n} tool"),
                    input_schema: r#"{"type":"object"}"#.to_string(),
                })
                .collect(),
            ..Self::default()
        }
    }

    /// Appends completions, returned one per `llm.complete` call.
    pub fn complete(mut self, c: impl IntoIterator<Item = Result<Completion, LlmError>>) -> Self {
        self.completions.extend(c);
        self
    }

    /// Queues the next result of calling `tool`. Unqueued calls answer
    /// `{"done": "<tool>"}`.
    pub fn result(mut self, tool: &str, r: Result<&str, ToolError>) -> Self {
        self.results
            .entry(tool.to_string())
            .or_default()
            .push_back(r.map(str::to_string));
        self
    }

    /// Sets a state value visible to the run.
    pub fn state(mut self, k: &str, v: &str) -> Self {
        self.state.insert(k.into(), v.into());
        self
    }
}

/// Store state of one instantiation: the script plus everything the guest
/// did through its imports.
pub struct TestHost {
    script: Script,
    /// Every completion request, in order.
    pub requests: Vec<CompletionRequest>,
    /// Every `mcp.call`, with arguments parsed as JSON (a string if not
    /// JSON) and the origin the guest passed.
    pub calls: Vec<(String, Value, CallOrigin)>,
    /// State after the run.
    pub state: HashMap<String, String>,
    /// Every `log.emit`.
    pub logs: Vec<(Level, String)>,
    wasi: WasiCtx,
    table: ResourceTable,
}

impl TestHost {
    fn new(mut script: Script) -> Self {
        let state = std::mem::take(&mut script.state);
        Self {
            script,
            requests: Vec::new(),
            calls: Vec::new(),
            state,
            logs: Vec::new(),
            wasi: sandbox_wasi_ctx(),
            table: ResourceTable::new(),
        }
    }
}

impl WasiView for TestHost {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

impl types::Host for TestHost {}

impl llm::Host for TestHost {
    fn complete(&mut self, req: CompletionRequest) -> Result<Completion, LlmError> {
        self.requests.push(req);
        self.script
            .completions
            .pop_front()
            .unwrap_or_else(|| Err(LlmError::Unavailable("script exhausted".into())))
    }
}

impl mcp::Host for TestHost {
    fn call(
        &mut self,
        tool: String,
        args: String,
        origin: CallOrigin,
    ) -> Result<String, ToolError> {
        let parsed = serde_json::from_str(&args).unwrap_or(Value::String(args));
        let result = self
            .script
            .results
            .get_mut(&tool)
            .and_then(VecDeque::pop_front)
            .unwrap_or_else(|| Ok(serde_json::json!({ "done": tool }).to_string()));
        self.calls.push((tool, parsed, origin));
        result
    }

    fn list_tools(&mut self) -> Result<Vec<ToolInfo>, ToolError> {
        Ok(self.script.tools.clone())
    }
}

impl state::Host for TestHost {
    fn get(&mut self, key: String) -> Option<String> {
        self.state.get(&key).cloned()
    }

    fn set(&mut self, key: String, value: String) {
        self.state.insert(key, value);
    }
}

impl log::Host for TestHost {
    fn emit(&mut self, level: Level, msg: String) {
        self.logs.push((level, msg));
    }
}

impl clock::Host for TestHost {
    fn now(&mut self) -> u64 {
        FIXED_NOW_MS
    }
}

/// Instantiates the runner component over `script` and calls
/// `run(spec, event)`. Returns the guest's result and the host afterwards.
///
/// # Panics
///
/// If the component fails to link or instantiate, or traps.
pub fn run(spec: &str, event: &str, script: Script) -> (Result<String, RunError>, TestHost) {
    let mut linker = Linker::new(engine());
    add_runner_wasi(&mut linker).expect("WASI links");
    Runner::add_to_linker::<TestHost, HasSelf<TestHost>>(&mut linker, |h| h)
        .expect("runner imports link");
    let mut store = Store::new(engine(), TestHost::new(script));
    let runner =
        Runner::instantiate(&mut store, runner_component(), &linker).expect("runner instantiates");
    let result = runner
        .call_run(&mut store, spec, event)
        .expect("run does not trap");
    (result, store.into_data())
}

/// Instantiates the `env-probe` fixture with [`sandbox_wasi_ctx`] and returns
/// what its `probe` export reports, as JSON.
///
/// # Panics
///
/// If the fixture fails to build, link or instantiate, or traps.
pub fn probe_sandbox() -> Value {
    probe_with(sandbox_wasi_ctx())
}

/// Instantiates the `env-probe` fixture with `wasi` instead of the sandbox.
/// Used as the positive control: it shows the probe does report what a
/// context exposes, so an empty report from the sandbox means something.
///
/// # Panics
///
/// As [`probe_sandbox`].
pub fn probe_with(wasi: WasiCtx) -> Value {
    let component =
        Component::from_file(engine(), env_probe_path()).expect("env-probe component loads");
    let mut linker = Linker::new(engine());
    // The fixture reads the filesystem on purpose, so it gets every WASI
    // interface; the sandbox is the context, not the linker.
    wasmtime_wasi::p2::add_to_linker_sync(&mut linker).expect("WASI links");
    let mut host = TestHost::new(Script::default());
    host.wasi = wasi;
    let mut store = Store::new(engine(), host);
    let probe =
        EnvProbe::instantiate(&mut store, &component, &linker).expect("env-probe instantiates");
    let out = probe.call_probe(&mut store).expect("probe does not trap");
    serde_json::from_str(&out).expect("probe output is JSON")
}
