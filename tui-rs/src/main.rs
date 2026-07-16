use std::{
    collections::HashMap,
    env, fs,
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table, Wrap},
    Frame, Terminal,
};
use rusqlite::{Connection, OpenFlags};

const STRATEGIES: &[(&str, &str)] = &[
    ("LLM", "llm"),
    ("Template", "template"),
    ("Factor mining", "factor_mining"),
];
const DATASETS: &[(&str, &str)] = &[
    ("All datasets", "all"),
    ("Price / volume", "pv"),
    ("Fundamental", "fundamental"),
    ("Analyst", "analyst"),
    ("Model", "model"),
    ("News", "news"),
    ("Option", "option"),
    ("Sentiment", "sentiment"),
    ("Social media", "socialmedia"),
];
const MARKETS: &[(&str, &str)] = &[
    ("Default market", "default"),
    ("USA", "USA"),
    ("China", "CHN"),
    ("Europe", "EUR"),
    ("Asia", "ASI"),
    ("Global", "GLB"),
];
const MAX_LOG_LINES: usize = 2_000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Strategy,
    Dataset,
    Market,
    Count,
    Batches,
    Idea,
}

impl Focus {
    const ALL: [Self; 6] = [
        Self::Strategy,
        Self::Dataset,
        Self::Market,
        Self::Count,
        Self::Batches,
        Self::Idea,
    ];

    fn next(self, backwards: bool) -> Self {
        let index = Self::ALL.iter().position(|item| *item == self).unwrap_or(0);
        let next = if backwards {
            (index + Self::ALL.len() - 1) % Self::ALL.len()
        } else {
            (index + 1) % Self::ALL.len()
        };
        Self::ALL[next]
    }
}

#[derive(Default)]
struct Stats {
    generated: i64,
    backtesting: i64,
    evaluated: i64,
    high_quality: i64,
    submitted: i64,
    failed: i64,
}

struct AlphaRow {
    id: i64,
    status: String,
    strategy: String,
    created_at: String,
    expression: String,
}

struct Dashboard {
    stats: Stats,
    recent: Vec<AlphaRow>,
    error: Option<String>,
    refreshed_at: Instant,
}

impl Default for Dashboard {
    fn default() -> Self {
        Self {
            stats: Stats::default(),
            recent: Vec::new(),
            error: None,
            refreshed_at: Instant::now(),
        }
    }
}

enum JobEvent {
    Line(String),
    Finished { label: String, success: bool },
}

struct App {
    focus: Focus,
    editing: bool,
    strategy: usize,
    dataset: usize,
    market: usize,
    count: String,
    batches: String,
    idea: String,
    dashboard: Dashboard,
    db_path: PathBuf,
    min_fitness: f64,
    logs: Vec<String>,
    log_scroll: u16,
    running: Option<String>,
    cancel_tx: Option<Sender<()>>,
    job_tx: Sender<JobEvent>,
    job_rx: Receiver<JobEvent>,
    quit_after_job: bool,
    should_quit: bool,
}

impl App {
    fn new() -> Self {
        let (job_tx, job_rx) = mpsc::channel();
        let db_path = setting("DB_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(default_db_path);
        let min_fitness = setting("MIN_FITNESS")
            .and_then(|value| value.parse().ok())
            .unwrap_or(1.0);
        let mut app = Self {
            focus: Focus::Strategy,
            editing: false,
            strategy: 0,
            dataset: 0,
            market: 0,
            count: "18".into(),
            batches: "1".into(),
            idea: String::new(),
            dashboard: Dashboard::default(),
            db_path,
            min_fitness,
            logs: vec!["AlphaGen Agent Rust TUI ready".into()],
            log_scroll: 0,
            running: None,
            cancel_tx: None,
            job_tx,
            job_rx,
            quit_after_job: false,
            should_quit: false,
        };
        app.refresh_dashboard();
        app
    }

    fn refresh_dashboard(&mut self) {
        match load_dashboard(&self.db_path, self.min_fitness) {
            Ok(dashboard) => self.dashboard = dashboard,
            Err(error) => {
                self.dashboard.error = Some(error.to_string());
                self.dashboard.refreshed_at = Instant::now();
            }
        }
    }

    fn poll_job(&mut self) -> bool {
        let mut changed = false;
        let mut finished = false;
        while let Ok(event) = self.job_rx.try_recv() {
            changed = true;
            match event {
                JobEvent::Line(line) => {
                    self.logs.push(line);
                    if self.logs.len() > MAX_LOG_LINES {
                        let remove = self.logs.len() - MAX_LOG_LINES;
                        self.logs.drain(..remove);
                    }
                    self.log_scroll = 0;
                }
                JobEvent::Finished { label, success } => {
                    let marker = if success { "OK" } else { "ERR" };
                    self.logs.push(format!("{marker} {label} finished"));
                    finished = true;
                }
            }
        }
        if finished {
            self.running = None;
            self.cancel_tx = None;
            self.refresh_dashboard();
            if self.quit_after_job {
                self.should_quit = true;
            }
        }
        changed
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if self.editing {
            self.handle_edit_key(key);
            return;
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('r') {
            self.refresh_dashboard();
            return;
        }

        match key.code {
            KeyCode::Char('q') => {
                if self.running.is_some() {
                    self.quit_after_job = true;
                    self.cancel_running();
                } else {
                    self.should_quit = true;
                }
            }
            KeyCode::Char('c') if self.running.is_some() => self.cancel_running(),
            KeyCode::Char('g') => self.start_job(JobKind::Generate),
            KeyCode::Char('r') => self.start_job(JobKind::Run),
            KeyCode::Char('f') => self.start_job(JobKind::Refine),
            KeyCode::Char('b') => self.start_job(JobKind::Backtest),
            KeyCode::Tab => self.focus = self.focus.next(false),
            KeyCode::BackTab => self.focus = self.focus.next(true),
            KeyCode::Up => self.change_selection(-1),
            KeyCode::Down => self.change_selection(1),
            KeyCode::Enter | KeyCode::Char('i') => {
                if matches!(self.focus, Focus::Count | Focus::Batches | Focus::Idea) {
                    self.editing = true;
                }
            }
            KeyCode::PageUp => self.log_scroll = self.log_scroll.saturating_add(8),
            KeyCode::PageDown => self.log_scroll = self.log_scroll.saturating_sub(8),
            _ => {}
        }
    }

    fn handle_edit_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc | KeyCode::Enter => self.editing = false,
            KeyCode::Backspace => {
                self.active_input().pop();
            }
            KeyCode::Char(ch) => {
                if matches!(self.focus, Focus::Count | Focus::Batches) && !ch.is_ascii_digit() {
                    return;
                }
                self.active_input().push(ch);
            }
            _ => {}
        }
    }

    fn active_input(&mut self) -> &mut String {
        match self.focus {
            Focus::Count => &mut self.count,
            Focus::Batches => &mut self.batches,
            Focus::Idea => &mut self.idea,
            _ => unreachable!("only text fields enter edit mode"),
        }
    }

    fn change_selection(&mut self, delta: isize) {
        let selection = match self.focus {
            Focus::Strategy => Some((&mut self.strategy, STRATEGIES.len())),
            Focus::Dataset => Some((&mut self.dataset, DATASETS.len())),
            Focus::Market => Some((&mut self.market, MARKETS.len())),
            _ => None,
        };
        if let Some((current, len)) = selection {
            *current = ((*current as isize + delta).rem_euclid(len as isize)) as usize;
        }
    }

    fn start_job(&mut self, kind: JobKind) {
        if self.running.is_some() {
            self.logs
                .push("A task is already running (press c to cancel).".into());
            return;
        }
        let label = kind.label().to_string();
        let args = self.job_args(kind);
        let tx = self.job_tx.clone();
        let (cancel_tx, cancel_rx) = mpsc::channel();
        self.logs.push(format!("> {label}"));
        self.running = Some(label.clone());
        self.cancel_tx = Some(cancel_tx);
        spawn_backend(label, args, tx, cancel_rx);
    }

    fn job_args(&self, kind: JobKind) -> Vec<String> {
        let mut args = match kind {
            JobKind::Generate => vec![
                "generate".into(),
                "--strategy".into(),
                STRATEGIES[self.strategy].1.into(),
                "--count".into(),
                positive_number(&self.count, "18"),
                "--no-backtest".into(),
            ],
            JobKind::Run => vec![
                "run".into(),
                "--strategy".into(),
                STRATEGIES[self.strategy].1.into(),
                "--count".into(),
                positive_number(&self.count, "18"),
                "--batches".into(),
                positive_number(&self.batches, "1"),
                "--interval".into(),
                "0".into(),
            ],
            JobKind::Refine => vec![
                "refine".into(),
                "--count".into(),
                positive_number(&self.count, "10"),
            ],
            JobKind::Backtest => vec!["backtest".into(), "--pending".into()],
        };
        if DATASETS[self.dataset].1 != "all" && !matches!(kind, JobKind::Backtest) {
            args.push("--dataset".into());
            args.push(DATASETS[self.dataset].1.into());
        }
        if MARKETS[self.market].1 != "default" {
            args.push("--region".into());
            args.push(MARKETS[self.market].1.into());
        }
        if !self.idea.trim().is_empty() && matches!(kind, JobKind::Generate | JobKind::Run) {
            args.push("--idea".into());
            args.push(self.idea.trim().into());
        }
        args
    }

    fn cancel_running(&mut self) {
        if let Some(tx) = self.cancel_tx.take() {
            let _ = tx.send(());
            self.logs.push("Cancelling task...".into());
        }
    }
}

#[derive(Clone, Copy)]
enum JobKind {
    Generate,
    Run,
    Refine,
    Backtest,
}

impl JobKind {
    fn label(self) -> &'static str {
        match self {
            Self::Generate => "generate",
            Self::Run => "run pipeline",
            Self::Refine => "refine",
            Self::Backtest => "backtest pending",
        }
    }
}

fn main() -> Result<()> {
    enable_raw_mode().context("enable terminal raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("create terminal")?;

    let result = run(&mut terminal);

    disable_raw_mode().context("disable terminal raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen).context("leave alternate screen")?;
    terminal.show_cursor().context("restore cursor")?;
    result
}

fn run(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    let mut app = App::new();
    let tick = Duration::from_millis(50);
    let mut dirty = true;
    while !app.should_quit {
        dirty |= app.poll_job();
        if app.running.is_some() && app.dashboard.refreshed_at.elapsed() >= Duration::from_secs(2) {
            app.refresh_dashboard();
            dirty = true;
        }
        if dirty {
            terminal.draw(|frame| draw(frame, &app))?;
            dirty = false;
        }
        if event::poll(tick)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    app.handle_key(key);
                    dirty = true;
                }
            }
        }
    }
    Ok(())
}

fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(12),
            Constraint::Length(2),
        ])
        .split(area);

    draw_header(frame, app, outer[0]);
    let content = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(34), Constraint::Min(50)])
        .split(outer[1]);
    draw_sidebar(frame, app, content[0]);
    draw_main(frame, app, content[1]);
    draw_footer(frame, app, outer[2]);
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let task = app.running.as_deref().unwrap_or("idle");
    let title = Line::from(vec![
        Span::styled(
            " AlphaGen Agent ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("Rust TUI", Style::default().fg(Color::DarkGray)),
        Span::raw("  "),
        Span::styled(
            task,
            if app.running.is_some() {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::Green)
            },
        ),
    ]);
    frame.render_widget(
        Paragraph::new(title).block(Block::default().borders(Borders::ALL)),
        area,
    );
}

fn draw_sidebar(frame: &mut Frame, app: &App, area: Rect) {
    let fields = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(3),
        ])
        .split(area);
    frame.render_widget(
        Block::default().title(" Research ").borders(Borders::RIGHT),
        area,
    );
    field(
        frame,
        fields[0],
        "Strategy",
        STRATEGIES[app.strategy].0,
        app.focus == Focus::Strategy,
        false,
    );
    field(
        frame,
        fields[1],
        "Dataset",
        DATASETS[app.dataset].0,
        app.focus == Focus::Dataset,
        false,
    );
    field(
        frame,
        fields[2],
        "Market",
        MARKETS[app.market].0,
        app.focus == Focus::Market,
        false,
    );
    field(
        frame,
        fields[3],
        "Count",
        &app.count,
        app.focus == Focus::Count,
        app.editing,
    );
    field(
        frame,
        fields[4],
        "Batches",
        &app.batches,
        app.focus == Focus::Batches,
        app.editing,
    );
    field(
        frame,
        fields[5],
        "Idea",
        if app.idea.is_empty() {
            "(optional)"
        } else {
            &app.idea
        },
        app.focus == Focus::Idea,
        app.editing,
    );
}

fn field(frame: &mut Frame, area: Rect, label: &str, value: &str, focused: bool, editing: bool) {
    let border = if focused {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    let title = if focused && editing {
        format!(" {label} [editing] ")
    } else {
        format!(" {label} ")
    };
    let text_style = if value == "(optional)" {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
    };
    frame.render_widget(
        Paragraph::new(value).style(text_style).block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border)),
        ),
        area,
    );
}

fn draw_main(frame: &mut Frame, app: &App, area: Rect) {
    let panels = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Min(7),
            Constraint::Length(10),
        ])
        .split(area);

    let stats = &app.dashboard.stats;
    let status = if let Some(error) = &app.dashboard.error {
        Text::from(vec![
            Line::styled("Dashboard unavailable", Style::default().fg(Color::Red)),
            Line::raw(error),
        ])
    } else {
        Text::from(vec![
            Line::from(vec![
                Span::raw("generated: "),
                Span::styled(
                    stats.generated.to_string(),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw("  backtesting: "),
                Span::styled(
                    stats.backtesting.to_string(),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw("  evaluated: "),
                Span::styled(
                    stats.evaluated.to_string(),
                    Style::default().fg(Color::Blue),
                ),
            ]),
            Line::from(vec![
                Span::raw("high: "),
                Span::styled(
                    stats.high_quality.to_string(),
                    Style::default().fg(Color::Green),
                ),
                Span::raw("  submitted: "),
                Span::styled(
                    stats.submitted.to_string(),
                    Style::default().fg(Color::Magenta),
                ),
                Span::raw("  failed: "),
                Span::styled(stats.failed.to_string(), Style::default().fg(Color::Red)),
            ]),
            Line::styled(
                format!("db: {}", app.db_path.display()),
                Style::default().fg(Color::DarkGray),
            ),
        ])
    };
    frame.render_widget(
        Paragraph::new(status).block(Block::default().title(" Dashboard ").borders(Borders::ALL)),
        panels[0],
    );

    let visible_height = panels[1].height.saturating_sub(2) as usize;
    let end = app.logs.len().saturating_sub(app.log_scroll as usize);
    let start = end.saturating_sub(visible_height);
    let log_text = app.logs[start..end]
        .iter()
        .map(|line| Line::raw(line.as_str()))
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(log_text)
            .wrap(Wrap { trim: false })
            .block(Block::default().title(" Task log ").borders(Borders::ALL)),
        panels[1],
    );

    let rows = app.dashboard.recent.iter().map(|alpha| {
        Row::new(vec![
            Cell::from(alpha.id.to_string()),
            Cell::from(alpha.status.as_str()),
            Cell::from(alpha.strategy.as_str()),
            Cell::from(short_date(&alpha.created_at)),
            Cell::from(truncate(&alpha.expression, 72)),
        ])
    });
    let header = Row::new(["ID", "Status", "Strategy", "Created", "Expression"]).style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let table = Table::new(
        rows,
        [
            Constraint::Length(7),
            Constraint::Length(13),
            Constraint::Length(13),
            Constraint::Length(12),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .column_spacing(1)
    .block(
        Block::default()
            .title(" Recent alphas ")
            .borders(Borders::ALL),
    );
    frame.render_widget(table, panels[2]);
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    frame.render_widget(Clear, area);
    let help = if app.editing {
        " Enter/Esc finish editing"
    } else if app.running.is_some() {
        " g generate  r run  f refine  b backtest  c cancel  PgUp/PgDn logs  q quit"
    } else {
        " Tab fields  ↑/↓ select  Enter/i edit  g generate  r run  f refine  b backtest  Ctrl+R refresh  q quit"
    };
    frame.render_widget(
        Paragraph::new(help).style(Style::default().fg(Color::DarkGray)),
        area,
    );
}

fn load_dashboard(path: &Path, min_fitness: f64) -> Result<Dashboard> {
    if !path.exists() {
        anyhow::bail!("database does not exist yet: {}", path.display());
    }
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("open {}", path.display()))?;
    conn.busy_timeout(Duration::from_millis(150))?;

    let mut counts = HashMap::new();
    let mut statement = conn.prepare("SELECT status, COUNT(*) FROM alphas GROUP BY status")?;
    for row in statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })? {
        let (status, count) = row?;
        counts.insert(status, count);
    }

    let high_quality = conn.query_row(
        "WITH latest AS (
             SELECT alpha_id, MAX(created_at) AS created_at
             FROM backtest_results GROUP BY alpha_id
         )
         SELECT COUNT(*)
         FROM backtest_results b
         JOIN latest l ON l.alpha_id = b.alpha_id AND l.created_at = b.created_at
         WHERE b.fitness >= ?1",
        [min_fitness],
        |row| row.get(0),
    )?;

    let mut recent_statement = conn.prepare(
        "SELECT id, status, strategy, created_at, expression
         FROM alphas ORDER BY created_at DESC LIMIT 12",
    )?;
    let recent = recent_statement
        .query_map([], |row| {
            Ok(AlphaRow {
                id: row.get(0)?,
                status: row.get(1)?,
                strategy: row.get(2)?,
                created_at: row.get(3)?,
                expression: row.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    Ok(Dashboard {
        stats: Stats {
            generated: *counts.get("generated").unwrap_or(&0),
            backtesting: *counts.get("backtesting").unwrap_or(&0),
            evaluated: *counts.get("evaluated").unwrap_or(&0),
            high_quality,
            submitted: *counts.get("submitted").unwrap_or(&0),
            failed: *counts.get("failed").unwrap_or(&0),
        },
        recent,
        error: None,
        refreshed_at: Instant::now(),
    })
}

fn spawn_backend(label: String, args: Vec<String>, tx: Sender<JobEvent>, cancel_rx: Receiver<()>) {
    thread::spawn(move || {
        let python = env::var("ALPHAGEN_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut command = Command::new(python);
        command.args(["-m", "alphagen_agent.cli"]);
        command.args(args);
        command.env("NO_COLOR", "1");
        command.env("PYTHONUNBUFFERED", "1");
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = tx.send(JobEvent::Line(format!(
                    "Failed to start Python backend: {error}"
                )));
                let _ = tx.send(JobEvent::Finished {
                    label,
                    success: false,
                });
                return;
            }
        };

        let mut readers = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            readers.push(spawn_reader(stdout, tx.clone()));
        }
        if let Some(stderr) = child.stderr.take() {
            readers.push(spawn_reader(stderr, tx.clone()));
        }

        let success = loop {
            match cancel_rx.try_recv() {
                Ok(()) | Err(TryRecvError::Disconnected) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = tx.send(JobEvent::Line("Task cancelled.".into()));
                    break false;
                }
                Err(TryRecvError::Empty) => {}
            }
            match child.try_wait() {
                Ok(Some(status)) => break status.success(),
                Ok(None) => thread::sleep(Duration::from_millis(50)),
                Err(error) => {
                    let _ = tx.send(JobEvent::Line(format!("Backend wait failed: {error}")));
                    break false;
                }
            }
        };
        for reader in readers {
            let _ = reader.join();
        }
        let _ = tx.send(JobEvent::Finished { label, success });
    });
}

fn spawn_reader<R: io::Read + Send + 'static>(
    stream: R,
    tx: Sender<JobEvent>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        for line in BufReader::new(stream).lines().map_while(|line| line.ok()) {
            let clean = strip_ansi(&line);
            if !clean.trim().is_empty() {
                let _ = tx.send(JobEvent::Line(clean));
            }
        }
    })
}

fn setting(key: &str) -> Option<String> {
    env::var(key).ok().or_else(|| {
        let text = fs::read_to_string(".env").ok()?;
        text.lines().find_map(|line| {
            let line = line.trim_start_matches('\u{feff}').trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (name, value) = line.split_once('=')?;
            (name.trim() == key).then(|| value.trim().trim_matches(['\'', '"']).to_string())
        })
    })
}

fn default_db_path() -> PathBuf {
    let current = PathBuf::from("alphagen_agent.db");
    let legacy = PathBuf::from("wq_agent.db");
    if !current.exists() && legacy.exists() {
        legacy
    } else {
        current
    }
}

fn positive_number(value: &str, fallback: &str) -> String {
    match value.parse::<u32>() {
        Ok(number) if number > 0 => number.to_string(),
        _ => fallback.into(),
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.into();
    }
    let mut text = value
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    text.push_str("...");
    text
}

fn short_date(value: &str) -> String {
    if value.len() >= 16 {
        value[5..16].replace('T', " ")
    } else {
        value.into()
    }
}

fn strip_ansi(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if ('@'..='~').contains(&next) {
                    break;
                }
            }
        } else {
            output.push(ch);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_unicode_by_character() {
        assert_eq!(truncate("abcdef", 8), "abcdef");
        assert_eq!(truncate("abcdefghij", 8), "abcde...");
        assert_eq!(truncate("中文表达式很长", 6), "中文表...");
    }

    #[test]
    fn strips_terminal_colors() {
        assert_eq!(strip_ansi("\u{1b}[31mError\u{1b}[0m"), "Error");
    }

    #[test]
    fn validates_positive_numbers() {
        assert_eq!(positive_number("4", "1"), "4");
        assert_eq!(positive_number("0", "1"), "1");
        assert_eq!(positive_number("bad", "18"), "18");
    }

    #[test]
    fn builds_backend_arguments_from_form_state() {
        let mut app = App::new();
        app.strategy = 2;
        app.dataset = 2;
        app.market = 1;
        app.count = "7".into();
        app.batches = "3".into();
        app.idea = "quality minus leverage".into();

        assert_eq!(
            app.job_args(JobKind::Run),
            vec![
                "run",
                "--strategy",
                "factor_mining",
                "--count",
                "7",
                "--batches",
                "3",
                "--interval",
                "0",
                "--dataset",
                "fundamental",
                "--region",
                "USA",
                "--idea",
                "quality minus leverage",
            ]
        );
    }
}
