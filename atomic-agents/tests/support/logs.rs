//! Captures every `tracing` event emitted in the harness process, so
//! scenarios can assert what rmcp itself logs.

use std::io::Write;
use std::sync::{Mutex, OnceLock};

static BUFFER: OnceLock<Mutex<Vec<u8>>> = OnceLock::new();

fn buffer() -> &'static Mutex<Vec<u8>> {
    BUFFER.get_or_init(Mutex::default)
}

struct Sink;

impl Write for Sink {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        buffer().lock().unwrap().extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Installs a global subscriber recording events at every level.
pub fn install() {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(|| Sink)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("install log capture");
}

/// The current end of the captured log, for [`since`].
pub fn mark() -> usize {
    buffer().lock().unwrap().len()
}

/// Everything logged after `mark`.
pub fn since(mark: usize) -> String {
    String::from_utf8_lossy(&buffer().lock().unwrap()[mark..]).into_owned()
}
