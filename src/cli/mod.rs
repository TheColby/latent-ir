use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "latent-ir")]
#[command(about = "Generative acoustics for the command line")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Generate a new synthetic impulse response.
    Generate(GenerateArgs),
    /// Analyze an impulse response WAV file.
    Analyze(AnalyzeArgs),
    /// Morph between two impulse responses.
    Morph(MorphArgs),
    /// Render audio through an impulse response.
    Render(RenderArgs),
    /// Output random descriptor samples.
    Sample(SampleArgs),
    /// List or inspect built-in presets.
    Preset(PresetArgs),
}

#[derive(Debug, Clone, Args)]
pub struct GenerateArgs {
    /// Optional semantic text prompt.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Optional built-in preset name.
    #[arg(long)]
    pub preset: Option<String>,

    /// Output WAV path.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Optional metadata JSON output path (defaults to companion .json).
    #[arg(long)]
    pub metadata_out: Option<PathBuf>,

    /// Optional analysis-only JSON output path.
    #[arg(long)]
    pub json_analysis_out: Option<PathBuf>,

    /// Sample rate for generated IR.
    #[arg(long, default_value_t = 48_000)]
    pub sample_rate: u32,

    /// RNG seed for deterministic generation.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    #[arg(long)]
    pub duration: Option<f32>,

    #[arg(long)]
    pub t60: Option<f32>,

    #[arg(long)]
    pub predelay_ms: Option<f32>,

    #[arg(long)]
    pub edt: Option<f32>,

    #[arg(long)]
    pub brightness: Option<f32>,

    #[arg(long)]
    pub diffusion: Option<f32>,

    #[arg(long)]
    pub early_density: Option<f32>,

    #[arg(long)]
    pub late_density: Option<f32>,

    #[arg(long)]
    pub width: Option<f32>,

    #[arg(long)]
    pub decorrelation: Option<f32>,

    #[arg(long, value_enum, default_value_t = ChannelFormatArg::Stereo)]
    pub channels: ChannelFormatArg,
}

#[derive(Debug, Clone, Args)]
pub struct AnalyzeArgs {
    /// Input IR WAV path.
    pub input: PathBuf,

    /// Emit JSON report.
    #[arg(long)]
    pub json: bool,

    /// Optional JSON output path.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct MorphArgs {
    /// First IR WAV file.
    pub ir_a: PathBuf,

    /// Second IR WAV file.
    pub ir_b: PathBuf,

    /// Interpolation amount [0,1].
    #[arg(long, default_value_t = 0.5)]
    pub alpha: f32,

    /// Output WAV path.
    #[arg(short, long)]
    pub output: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct RenderArgs {
    /// Dry input WAV file.
    pub input: PathBuf,

    /// IR WAV file.
    #[arg(long)]
    pub ir: PathBuf,

    /// Wet/dry mix [0,1], where 1.0 is fully wet.
    #[arg(long, default_value_t = 0.25)]
    pub mix: f32,

    /// Output WAV path.
    #[arg(short, long)]
    pub output: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct SampleArgs {
    /// Number of random descriptor sets to print.
    #[arg(long, default_value_t = 4)]
    pub count: usize,

    /// Seed for reproducible sampling.
    #[arg(long, default_value_t = 7)]
    pub seed: u64,

    /// Emit JSON output.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Clone, Args)]
pub struct PresetArgs {
    /// Preset name for details. Omit to list all presets.
    pub name: Option<String>,

    /// Emit JSON output.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ChannelFormatArg {
    Mono,
    Stereo,
}
