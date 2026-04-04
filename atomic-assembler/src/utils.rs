use std::path::{Path, PathBuf};

const GITHUB_URL: &str = "https://github.com/BrainBlend-AI/atomic-agents.git";

/// Clones the atomic-agents repository to a temporary directory.
pub fn clone_repo() -> Result<PathBuf, String> {
    let temp_dir = std::env::temp_dir().join("atomic-assembler-repo");

    // If already cloned, try to pull
    if temp_dir.exists() {
        // Just reuse the existing clone
        return Ok(temp_dir);
    }

    // Clone fresh
    git2::Repository::clone(GITHUB_URL, &temp_dir)
        .map_err(|e| format!("Failed to clone repository: {e}"))?;

    Ok(temp_dir)
}

/// Copies a tool from the repository to a destination directory.
///
/// Skips build/config files (requirements.txt, pyproject.toml, .coveragerc, uv.lock).
pub fn copy_tool(tool_path: &Path, dest: &Path) -> Result<(), String> {
    let tool_name = tool_path
        .file_name()
        .ok_or("Invalid tool path")?
        .to_string_lossy()
        .to_string();

    let dest_dir = dest.join(&tool_name);
    std::fs::create_dir_all(&dest_dir)
        .map_err(|e| format!("Failed to create destination directory: {e}"))?;

    copy_dir_recursive(tool_path, &dest_dir)?;

    Ok(())
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), String> {
    let skip_files = [
        "requirements.txt",
        "pyproject.toml",
        ".coveragerc",
        "uv.lock",
        "__pycache__",
        ".git",
    ];

    let entries = std::fs::read_dir(src)
        .map_err(|e| format!("Failed to read directory {}: {e}", src.display()))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
        let file_name = entry.file_name().to_string_lossy().to_string();

        if skip_files.contains(&file_name.as_str()) {
            continue;
        }

        let src_path = entry.path();
        let dest_path = dest.join(&file_name);

        if src_path.is_dir() {
            std::fs::create_dir_all(&dest_path)
                .map_err(|e| format!("Failed to create directory: {e}"))?;
            copy_dir_recursive(&src_path, &dest_path)?;
        } else {
            std::fs::copy(&src_path, &dest_path)
                .map_err(|e| format!("Failed to copy file: {e}"))?;
        }
    }

    Ok(())
}
