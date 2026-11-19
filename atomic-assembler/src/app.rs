use crossterm::event::KeyCode;
use ratatui::prelude::*;

use crate::screens::{
    Screen, file_explorer::FileExplorerScreen, main_menu::MainMenuScreen,
    tool_explorer::ToolExplorerScreen, tool_info::ToolInfoScreen,
};

/// The active screen in the application.
#[derive(Clone, Debug)]
pub enum ActiveScreen {
    MainMenu,
    ToolExplorer,
    FileExplorer {
        tool_name: String,
        tool_path: std::path::PathBuf,
    },
    ToolInfo {
        tool_name: String,
        readme: String,
    },
}

/// Main application state.
pub struct App {
    pub active_screen: ActiveScreen,
    pub main_menu: MainMenuScreen,
    pub tool_explorer: ToolExplorerScreen,
    pub file_explorer: FileExplorerScreen,
    pub tool_info: ToolInfoScreen,
    pub status_message: Option<String>,
}

impl App {
    pub fn new() -> Self {
        Self {
            active_screen: ActiveScreen::MainMenu,
            main_menu: MainMenuScreen::new(),
            tool_explorer: ToolExplorerScreen::new(),
            file_explorer: FileExplorerScreen::new(),
            tool_info: ToolInfoScreen::new(),
            status_message: None,
        }
    }

    pub fn render(&self, frame: &mut ratatui::Frame) {
        match &self.active_screen {
            ActiveScreen::MainMenu => self.main_menu.render(frame, frame.area()),
            ActiveScreen::ToolExplorer => self.tool_explorer.render(frame, frame.area()),
            ActiveScreen::FileExplorer { .. } => self.file_explorer.render(frame, frame.area()),
            ActiveScreen::ToolInfo { .. } => self.tool_info.render(frame, frame.area()),
        }

        // Status bar
        if let Some(msg) = &self.status_message {
            let area = frame.area();
            let status_area = Rect::new(0, area.height.saturating_sub(1), area.width, 1);
            let status = ratatui::widgets::Paragraph::new(msg.as_str())
                .style(Style::default().fg(Color::Yellow));
            frame.render_widget(status, status_area);
        }
    }

    /// Handles a key event. Returns true if the app should exit.
    pub fn handle_key(&mut self, key: KeyCode) -> bool {
        match &self.active_screen {
            ActiveScreen::MainMenu => match self.main_menu.handle_key(key) {
                Some(crate::screens::main_menu::MenuAction::DownloadTools) => {
                    self.tool_explorer.load_tools();
                    self.active_screen = ActiveScreen::ToolExplorer;
                    false
                }
                Some(crate::screens::main_menu::MenuAction::OpenGithub) => {
                    let _ = open::that("https://github.com/Eigenwise/atomic-agents");
                    false
                }
                Some(crate::screens::main_menu::MenuAction::Quit) => true,
                None => false,
            },
            ActiveScreen::ToolExplorer => {
                match self.tool_explorer.handle_key(key) {
                    Some(crate::screens::tool_explorer::ExplorerAction::Back) => {
                        self.active_screen = ActiveScreen::MainMenu;
                    }
                    Some(crate::screens::tool_explorer::ExplorerAction::SelectTool(name, path)) => {
                        self.file_explorer.set_mode_directory();
                        self.active_screen = ActiveScreen::FileExplorer {
                            tool_name: name,
                            tool_path: path,
                        };
                    }
                    Some(crate::screens::tool_explorer::ExplorerAction::ShowInfo(name, readme)) => {
                        self.tool_info.set_content(&name, &readme);
                        self.active_screen = ActiveScreen::ToolInfo {
                            tool_name: name,
                            readme,
                        };
                    }
                    None => {}
                }
                false
            }
            ActiveScreen::FileExplorer {
                tool_name,
                tool_path,
            } => {
                let tool_name = tool_name.clone();
                let tool_path = tool_path.clone();
                match self.file_explorer.handle_key(key) {
                    Some(crate::screens::file_explorer::FileAction::Select(dest)) => {
                        match crate::utils::copy_tool(&tool_path, &dest) {
                            Ok(_) => {
                                self.status_message = Some(format!(
                                    "✓ Tool '{}' copied to {}",
                                    tool_name,
                                    dest.display()
                                ));
                            }
                            Err(e) => {
                                self.status_message = Some(format!("✗ Failed to copy tool: {e}"));
                            }
                        }
                        self.active_screen = ActiveScreen::ToolExplorer;
                    }
                    Some(crate::screens::file_explorer::FileAction::Back) => {
                        self.active_screen = ActiveScreen::ToolExplorer;
                    }
                    None => {}
                }
                false
            }
            ActiveScreen::ToolInfo { .. } => {
                if matches!(key, KeyCode::Esc) {
                    self.active_screen = ActiveScreen::ToolExplorer;
                }
                false
            }
        }
    }
}
