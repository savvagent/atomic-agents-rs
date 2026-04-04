pub mod file_explorer;
pub mod main_menu;
pub mod tool_explorer;
pub mod tool_info;

use crossterm::event::KeyCode;
use ratatui::prelude::*;

/// Common trait for TUI screens.
pub trait Screen {
    fn render(&self, frame: &mut ratatui::Frame, area: Rect);
}
