use std::path::PathBuf;

use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;

use super::Screen;

pub enum ExplorerAction {
    Back,
    SelectTool(String, PathBuf),
    ShowInfo(String, String),
}

#[derive(Clone)]
pub struct ToolEntry {
    pub name: String,
    pub path: PathBuf,
}

pub struct ToolExplorerScreen {
    tools: Vec<ToolEntry>,
    selected: usize,
    loading: bool,
    repo_path: Option<PathBuf>,
}

impl ToolExplorerScreen {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            selected: 0,
            loading: false,
            repo_path: None,
        }
    }

    pub fn load_tools(&mut self) {
        self.loading = true;
        // Try to clone the repo and list tools
        match crate::utils::clone_repo() {
            Ok(path) => {
                let tools_dir = path.join("atomic-forge").join("tools");
                if let Ok(entries) = std::fs::read_dir(&tools_dir) {
                    self.tools = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_dir())
                        .map(|e| ToolEntry {
                            name: e.file_name().to_string_lossy().to_string(),
                            path: e.path(),
                        })
                        .collect();
                    self.tools.sort_by(|a, b| a.name.cmp(&b.name));
                }
                self.repo_path = Some(path);
            }
            Err(e) => {
                tracing::error!("Failed to clone repo: {e}");
            }
        }
        self.loading = false;
        self.selected = 0;
    }

    pub fn handle_key(&mut self, key: KeyCode) -> Option<ExplorerAction> {
        match key {
            KeyCode::Esc | KeyCode::Char('q') => Some(ExplorerAction::Back),
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected < self.tools.len().saturating_sub(1) {
                    self.selected += 1;
                }
                None
            }
            KeyCode::Enter => {
                if let Some(tool) = self.tools.get(self.selected) {
                    Some(ExplorerAction::SelectTool(
                        tool.name.clone(),
                        tool.path.clone(),
                    ))
                } else {
                    None
                }
            }
            KeyCode::Char('i') => {
                if let Some(tool) = self.tools.get(self.selected) {
                    let readme_path = tool.path.join("README.md");
                    let readme = std::fs::read_to_string(&readme_path)
                        .unwrap_or_else(|_| "No README.md found.".to_string());
                    Some(ExplorerAction::ShowInfo(tool.name.clone(), readme))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl Screen for ToolExplorerScreen {
    fn render(&self, frame: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(2),
            ])
            .split(area);

        // Title
        let title = Paragraph::new(" Atomic Tool Explorer")
            .style(Style::default().fg(Color::Magenta).bold())
            .block(Block::default().borders(Borders::BOTTOM));
        frame.render_widget(title, chunks[0]);

        if self.loading {
            let loading = Paragraph::new("  Loading tools from GitHub...")
                .style(Style::default().fg(Color::Yellow));
            frame.render_widget(loading, chunks[1]);
        } else if self.tools.is_empty() {
            let empty = Paragraph::new("  No tools found.").style(Style::default().fg(Color::Red));
            frame.render_widget(empty, chunks[1]);
        } else {
            let items: Vec<ListItem> = self
                .tools
                .iter()
                .enumerate()
                .map(|(i, tool)| {
                    let prefix = if i == self.selected {
                        "▸ 🔧 "
                    } else {
                        "  🔧 "
                    };
                    let style = if i == self.selected {
                        Style::default().fg(Color::Yellow).bold()
                    } else {
                        Style::default().fg(Color::White)
                    };
                    ListItem::new(format!("{prefix}{}", tool.name)).style(style)
                })
                .collect();

            let list = List::new(items).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(format!(" {} tools available ", self.tools.len()))
                    .border_style(Style::default().fg(Color::Magenta)),
            );
            frame.render_widget(list, chunks[1]);
        }

        // Help bar
        let help = Paragraph::new(" ↑↓ Navigate  Enter Download  i Info  Esc Back")
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(help, chunks[2]);
    }
}
