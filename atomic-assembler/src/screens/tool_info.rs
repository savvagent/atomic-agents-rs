use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;

use super::Screen;

pub struct ToolInfoScreen {
    tool_name: String,
    content: String,
    scroll: u16,
}

impl ToolInfoScreen {
    pub fn new() -> Self {
        Self {
            tool_name: String::new(),
            content: String::new(),
            scroll: 0,
        }
    }

    pub fn set_content(&mut self, name: &str, content: &str) {
        self.tool_name = name.to_string();
        self.content = content.to_string();
        self.scroll = 0;
    }

    pub fn handle_key(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Esc | KeyCode::Char('q') => true,
            KeyCode::Up | KeyCode::Char('k') => {
                self.scroll = self.scroll.saturating_sub(1);
                false
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.scroll = self.scroll.saturating_add(1);
                false
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_sub(10);
                false
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_add(10);
                false
            }
            _ => false,
        }
    }
}

impl Screen for ToolInfoScreen {
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
        let title = Paragraph::new(format!(" 🔧 {} - README", self.tool_name))
            .style(Style::default().fg(Color::Magenta).bold())
            .block(Block::default().borders(Borders::BOTTOM));
        frame.render_widget(title, chunks[0]);

        // Content
        let content = Paragraph::new(self.content.as_str())
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Tool Information ")
                    .border_style(Style::default().fg(Color::Magenta)),
            )
            .wrap(Wrap { trim: false })
            .scroll((self.scroll, 0));
        frame.render_widget(content, chunks[1]);

        // Help bar
        let help = Paragraph::new(" ↑↓/PgUp/PgDn Scroll  Esc Back")
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(help, chunks[2]);
    }
}
