use std::fmt::Display;
use std::io::{stderr, stdout, IsTerminal};

fn color_enabled_stdout() -> bool {
    stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

fn color_enabled_stderr() -> bool {
    stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

fn paint(s: &str, code: &str, enabled: bool) -> String {
    if enabled {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

pub fn section(title: &str) -> String {
    paint(title, "1;35", color_enabled_stdout())
}

pub fn metric(name: &str, value: impl Display) -> String {
    format!(
        "{}: {}",
        paint(name, "1;36", color_enabled_stdout()),
        paint(&format!("{value}"), "1;32", color_enabled_stdout())
    )
}

pub fn info(name: &str, value: impl Display) -> String {
    format!(
        "{}: {}",
        paint(name, "1;34", color_enabled_stdout()),
        paint(&format!("{value}"), "0;37", color_enabled_stdout())
    )
}

pub fn warning(msg: &str) -> String {
    paint(msg, "1;33", color_enabled_stdout())
}

pub fn success(msg: &str) -> String {
    paint(msg, "1;32", color_enabled_stdout())
}

pub fn error(msg: &str) -> String {
    paint(msg, "1;31", color_enabled_stderr())
}
