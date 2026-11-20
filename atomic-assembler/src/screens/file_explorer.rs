use std::path::PathBuf;

use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;

use super::Screen;

pub enum FileAction {
    Select(PathBuf),
    Back,
}

pub struct FileExplorerScreen {
    current_dir: PathBuf,
    entries: Vec<PathBuf>,
    selected: usize,
    directory_mode: bool,
}

impl FileExplorerScreen {
    pub fn new() -> Self {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        let mut screen = Self {
            current_dir: current_dir.clone(),
            entries: Vec::new(),
            selected: 0,
            directory_mode: false,
        };
        screen.refresh_entries();
        screen
    }

    pub fn set_mode_directory(&mut self) {
        self.directory_mode = true;
        self.current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        self.refresh_entries();
    }

    fn refresh_entries(&mut self) {
        self.entries.clear();
        if let Ok(entries) = std::fs::read_dir(&self.current_dir) {
            let mut dirs: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    if self.directory_mode {
                        e.path().is_dir()
                    } else {
                        true
                    }
                })
                .filter(|e| {
                    // Hide hidden files
                    !e.file_name().to_string_lossy().starts_with('.')
                })
                .map(|e| e.path())
                .collect();
            dirs.sort();
            self.entries = dirs;
        }
        self.selected = 0;
    }

    pub fn handle_key(&mut self, key: KeyCode) -> Option<FileAction> {
        match key {
            KeyCode::Esc | KeyCode::Char('q') => Some(FileAction::Back),
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected < self.entries.len().saturating_sub(1) {
                    self.selected += 1;
                }
                None
            }
            KeyCode::Enter => {
                if self.directory_mode {
                    // In directory mode, Enter selects the current directory
                    Some(FileAction::Select(self.current_dir.clone()))
                } else if let Some(path) = self.entries.get(self.selected) {
                    if path.is_dir() {
                        self.current_dir = path.clone();
                        self.refresh_entries();
                        None
                    } else {
                        Some(FileAction::Select(path.clone()))
                    }
                } else {
                    None
                }
            }
            KeyCode::Right | KeyCode::Char('l') => {
                if let Some(path) = self.entries.get(self.selected) {
                    if path.is_dir() {
                        self.current_dir = path.clone();
                        self.refresh_entries();
                    }
                }
                None
            }
            KeyCode::Left | KeyCode::Char('h') => {
                if let Some(parent) = self.current_dir.parent() {
                    self.current_dir = parent.to_path_buf();
                    self.refresh_entries();
                }
                None
            }
            _ => None,
        }
    }
}

impl Screen for FileExplorerScreen {
    fn render(&self, frame: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(2),
            ])
            .split(area);

        // Current path
        let path_display = format!(" 📁 {}", self.current_dir.display());
        let title = Paragraph::new(path_display)
            .style(Style::default().fg(Color::Cyan).bold())
            .block(Block::default().borders(Borders::BOTTOM));
        frame.render_widget(title, chunks[0]);

        // Entries
        let items: Vec<ListItem> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, path)| {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let icon = if path.is_dir() { "📁" } else { "📄" };
                let prefix = if i == self.selected { "▸ " } else { "  " };
                let style = if i == self.selected {
                    Style::default().fg(Color::Yellow).bold()
                } else {
                    Style::default().fg(Color::White)
                };
                ListItem::new(format!("{prefix}{icon} {name}")).style(style)
            })
            .collect();

        let title = if self.directory_mode {
            " Select destination directory "
        } else {
            " Browse files "
        };
        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(Style::default().fg(Color::Magenta)),
        );
        frame.render_widget(list, chunks[1]);

        // Help bar
        let help_text = if self.directory_mode {
            " ↑↓ Navigate  ←→ Parent/Enter dir  Enter Select this dir  Esc Back"
        } else {
            " ↑↓ Navigate  ←→ Parent/Enter dir  Enter Select  Esc Back"
        };
        let help = Paragraph::new(help_text).style(Style::default().fg(Color::DarkGray));
        frame.render_widget(help, chunks[2]);
    }
}
