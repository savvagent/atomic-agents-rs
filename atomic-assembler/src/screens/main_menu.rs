use crossterm::event::KeyCode;
use ratatui::prelude::*;
use ratatui::widgets::*;

use super::Screen;

pub enum MenuAction {
    DownloadTools,
    OpenGithub,
    Quit,
}

const MENU_ITEMS: &[&str] = &["Download Tools", "Open Atomic Agents on GitHub", "Quit"];

pub struct MainMenuScreen {
    selected: usize,
}

impl MainMenuScreen {
    pub fn new() -> Self {
        Self { selected: 0 }
    }

    pub fn handle_key(&mut self, key: KeyCode) -> Option<MenuAction> {
        match key {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected < MENU_ITEMS.len() - 1 {
                    self.selected += 1;
                }
                None
            }
            KeyCode::Enter => match self.selected {
                0 => Some(MenuAction::DownloadTools),
                1 => Some(MenuAction::OpenGithub),
                2 => Some(MenuAction::Quit),
                _ => None,
            },
            KeyCode::Char('q') | KeyCode::Esc => Some(MenuAction::Quit),
            _ => None,
        }
    }
}

impl Screen for MainMenuScreen {
    fn render(&self, frame: &mut ratatui::Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),
                Constraint::Min(0),
                Constraint::Length(2),
            ])
            .split(area);

        // Title
        let title = Paragraph::new(vec![
            Line::from(""),
            Line::from(
                "  ⚛  Atomic Assembler"
                    .to_string(),
            )
            .style(Style::default().fg(Color::Magenta).bold()),
            Line::from(""),
        ])
        .block(Block::default().borders(Borders::BOTTOM));
        frame.render_widget(title, chunks[0]);

        // Menu items
        let items: Vec<ListItem> = MENU_ITEMS
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let prefix = if i == self.selected { "▸ " } else { "  " };
                let style = if i == self.selected {
                    Style::default().fg(Color::Yellow).bold()
                } else {
                    Style::default().fg(Color::White)
                };
                ListItem::new(format!("{prefix}{item}")).style(style)
            })
            .collect();

        let menu = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Menu ")
                    .border_style(Style::default().fg(Color::Magenta)),
            );
        frame.render_widget(menu, chunks[1]);

        // Help bar
        let help = Paragraph::new(" ↑↓ Navigate  Enter Select  q Quit")
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(help, chunks[2]);
    }
}
