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
    /// Train lightweight learned conditioning encoders from labeled data.
    TrainEncoder(TrainEncoderArgs),
    /// Evaluate learned encoders against labeled datasets and emit baseline reports.
    Eval(EvalArgs),
    /// Benchmark models/pipelines and emit regression-friendly reports.
    Benchmark(BenchmarkArgs),
    /// Validate model manifests and runtime compatibility.
    Model(ModelArgs),
    /// Run one-shot A/B generation + analysis (industrial model vs baseline).
    AbTest(AbTestArgs),
}

#[derive(Debug, Clone, Args)]
pub struct GenerateArgs {
    /// Optional semantic text prompt.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Path to a learned text encoder model JSON.
    #[arg(long)]
    pub text_encoder_model: Option<PathBuf>,

    /// Path to a learned ONNX text encoder model (requires `onnx` feature).
    #[arg(long)]
    pub text_encoder_onnx: Option<PathBuf>,

    /// Input feature dimension for ONNX text encoder hashing frontend.
    #[arg(long, default_value_t = 256)]
    pub text_encoder_onnx_input_dim: usize,

    /// Path to reference audio used by learned audio conditioning.
    #[arg(long)]
    pub reference_audio: Option<PathBuf>,

    /// Path to a learned audio encoder model JSON.
    #[arg(long)]
    pub audio_encoder_model: Option<PathBuf>,

    /// Path to a learned ONNX audio encoder model (requires `onnx` feature).
    #[arg(long)]
    pub audio_encoder_onnx: Option<PathBuf>,

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

    /// Perceptual macro: perceived size [-1, 1].
    #[arg(long)]
    pub macro_size: Option<f32>,

    /// Perceptual macro: source/listener distance [-1, 1].
    #[arg(long)]
    pub macro_distance: Option<f32>,

    /// Perceptual macro: material hardness/brightness [-1, 1].
    #[arg(long)]
    pub macro_material: Option<f32>,

    /// Perceptual macro: clarity [-1, 1].
    #[arg(long)]
    pub macro_clarity: Option<f32>,

    /// Optional macro automation trajectory JSON path.
    #[arg(long)]
    pub macro_trajectory: Option<PathBuf>,

    /// Custom channel layout JSON path (required with `--channels custom`).
    #[arg(long)]
    pub layout_json: Option<PathBuf>,

    /// Optional channel-map JSON output path (defaults to companion `.channels.json`).
    #[arg(long)]
    pub channel_map_out: Option<PathBuf>,

    /// Output channel format (`mono`, `stereo`, `foa`, `5.1`, `7.1`, `7.1.4`, `7.2.4`, `custom`).
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

    /// Optional channel-map JSON path; defaults to companion `.channels.json` when present.
    #[arg(long)]
    pub channel_map: Option<PathBuf>,
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

    /// Rendering engine.
    #[arg(long, value_enum, default_value_t = RenderEngineArg::Auto)]
    pub engine: RenderEngineArg,

    /// Partition size for FFT partitioned convolution.
    #[arg(long, default_value_t = 2048)]
    pub partition_size: usize,

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

#[derive(Debug, Clone, Args)]
pub struct TrainEncoderArgs {
    #[command(subcommand)]
    pub mode: TrainEncoderMode,
}

#[derive(Debug, Clone, Subcommand)]
pub enum TrainEncoderMode {
    /// Train a prompt->descriptor text encoder model.
    Text(TrainTextEncoderArgs),
    /// Train a reference-audio->descriptor audio encoder model.
    Audio(TrainAudioEncoderArgs),
}

#[derive(Debug, Clone, Args)]
pub struct TrainTextEncoderArgs {
    /// Path to JSON dataset: [{\"prompt\":...,\"descriptor\":...}, ...]
    #[arg(long)]
    pub dataset: PathBuf,
    /// Output model JSON path.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Max vocabulary size.
    #[arg(long, default_value_t = 256)]
    pub max_vocab: usize,
    /// Minimum token count to keep in vocabulary.
    #[arg(long, default_value_t = 1)]
    pub min_count: usize,
    /// Number of optimization epochs.
    #[arg(long, default_value_t = 600)]
    pub epochs: usize,
    /// Learning rate.
    #[arg(long, default_value_t = 0.05)]
    pub lr: f32,
    /// L2 regularization weight.
    #[arg(long, default_value_t = 1e-4)]
    pub l2: f32,
}

#[derive(Debug, Clone, Args)]
pub struct TrainAudioEncoderArgs {
    /// Path to JSON dataset: [{\"audio_path\":...,\"descriptor\":...}, ...]
    #[arg(long)]
    pub dataset: PathBuf,
    /// Output model JSON path.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Number of optimization epochs.
    #[arg(long, default_value_t = 800)]
    pub epochs: usize,
    /// Learning rate.
    #[arg(long, default_value_t = 0.03)]
    pub lr: f32,
    /// L2 regularization weight.
    #[arg(long, default_value_t = 1e-4)]
    pub l2: f32,
}

#[derive(Debug, Clone, Args)]
pub struct EvalArgs {
    #[command(subcommand)]
    pub mode: EvalMode,
}

#[derive(Debug, Clone, Subcommand)]
pub enum EvalMode {
    /// Evaluate prompt-conditioned text encoder.
    Text(EvalTextArgs),
    /// Evaluate reference-audio-conditioned audio encoder.
    Audio(EvalAudioArgs),
}

#[derive(Debug, Clone, Args)]
pub struct EvalTextArgs {
    /// Labeled JSON dataset: [{\"prompt\":...,\"descriptor\":...}, ...]
    #[arg(long)]
    pub dataset: PathBuf,
    /// Learned text encoder model JSON.
    #[arg(long)]
    pub model: PathBuf,
    /// Output baseline report JSON path.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Sample rate used for analysis-space validation synthesis.
    #[arg(long, default_value_t = 48_000)]
    pub sample_rate: u32,
    /// RNG seed for deterministic validation synthesis.
    #[arg(long, default_value_t = 1234)]
    pub seed: u64,
}

#[derive(Debug, Clone, Args)]
pub struct EvalAudioArgs {
    /// Labeled JSON dataset: [{\"audio_path\":...,\"descriptor\":...}, ...]
    #[arg(long)]
    pub dataset: PathBuf,
    /// Learned audio encoder model JSON.
    #[arg(long)]
    pub model: PathBuf,
    /// Output baseline report JSON path.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Sample rate used for analysis-space validation synthesis.
    #[arg(long, default_value_t = 48_000)]
    pub sample_rate: u32,
    /// RNG seed for deterministic validation synthesis.
    #[arg(long, default_value_t = 1234)]
    pub seed: u64,
}

#[derive(Debug, Clone, Args)]
pub struct BenchmarkArgs {
    #[command(subcommand)]
    pub mode: BenchmarkMode,
}

#[derive(Debug, Clone, Subcommand)]
pub enum BenchmarkMode {
    /// Run benchmark suite on a benchmark dataset.
    Run(BenchmarkRunArgs),
    /// Check a benchmark report against a baseline and fail on regressions.
    Check(BenchmarkCheckArgs),
}

#[derive(Debug, Clone, Args)]
pub struct BenchmarkRunArgs {
    /// Benchmark dataset JSON path.
    #[arg(long)]
    pub dataset: PathBuf,
    /// Output benchmark report JSON path.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Optional learned text encoder JSON model.
    #[arg(long)]
    pub text_model: Option<PathBuf>,
    /// Optional learned audio encoder JSON model.
    #[arg(long)]
    pub audio_model: Option<PathBuf>,
    /// Sample rate for synthesis/analysis passes.
    #[arg(long, default_value_t = 48_000)]
    pub sample_rate: u32,
    /// Deterministic seed.
    #[arg(long, default_value_t = 2026)]
    pub seed: u64,
    /// Repeats per sample for stability stats.
    #[arg(long, default_value_t = 3)]
    pub repeats: usize,
}

#[derive(Debug, Clone, Args)]
pub struct BenchmarkCheckArgs {
    /// Candidate benchmark report JSON path.
    #[arg(long)]
    pub report: PathBuf,
    /// Baseline benchmark report JSON path.
    #[arg(long)]
    pub baseline: PathBuf,
    /// Allowed relative regression threshold (e.g. 0.05 == 5%).
    #[arg(long, default_value_t = 0.05)]
    pub max_regression: f32,
}

#[derive(Debug, Clone, Args)]
pub struct ModelArgs {
    #[command(subcommand)]
    pub mode: ModelMode,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ModelMode {
    /// Validate a model manifest against runtime capabilities.
    Validate(ModelValidateArgs),
}

#[derive(Debug, Clone, Args)]
pub struct ModelValidateArgs {
    /// Model manifest JSON path.
    #[arg(long)]
    pub manifest: PathBuf,
}

#[derive(Debug, Clone, Args)]
pub struct AbTestArgs {
    /// Prompt used for both A and B runs.
    #[arg(long)]
    pub prompt: String,
    /// Industrial text encoder model used for variant A.
    #[arg(long)]
    pub industrial_text_model: PathBuf,
    /// Directory for A/B artifacts and report.
    #[arg(long)]
    pub output_dir: PathBuf,
    /// Optional explicit t60 override.
    #[arg(long)]
    pub t60: Option<f32>,
    /// Optional duration override.
    #[arg(long)]
    pub duration: Option<f32>,
    /// Optional predelay override.
    #[arg(long)]
    pub predelay_ms: Option<f32>,
    /// Optional EDT override.
    #[arg(long)]
    pub edt: Option<f32>,
    /// Optional preset name.
    #[arg(long)]
    pub preset: Option<String>,
    /// Sample rate for both runs.
    #[arg(long, default_value_t = 48_000)]
    pub sample_rate: u32,
    /// Seed for deterministic comparison.
    #[arg(long, default_value_t = 1337)]
    pub seed: u64,
    #[arg(long)]
    pub macro_size: Option<f32>,
    #[arg(long)]
    pub macro_distance: Option<f32>,
    #[arg(long)]
    pub macro_material: Option<f32>,
    #[arg(long)]
    pub macro_clarity: Option<f32>,
    #[arg(long)]
    pub macro_trajectory: Option<PathBuf>,
    /// Custom channel layout JSON path (required with `--channels custom`).
    #[arg(long)]
    pub layout_json: Option<PathBuf>,
    /// Output channel format (`mono`, `stereo`, `foa`, `5.1`, `7.1`, `7.1.4`, `7.2.4`, `custom`).
    #[arg(long, value_enum, default_value_t = ChannelFormatArg::Stereo)]
    pub channels: ChannelFormatArg,
    /// Write a markdown scorecard (`ab_test_report.md`) in output directory.
    #[arg(long)]
    pub markdown: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ChannelFormatArg {
    #[value(alias = "1.0")]
    Mono,
    #[value(alias = "2.0")]
    Stereo,
    #[value(
        name = "foa",
        alias = "foa-ambix",
        alias = "ambix",
        alias = "ambisonic"
    )]
    Foa,
    #[value(
        name = "5.1",
        alias = "surround-5.1",
        alias = "surround5.1",
        alias = "5_1"
    )]
    Surround5_1,
    #[value(
        name = "7.1",
        alias = "surround-7.1",
        alias = "surround7.1",
        alias = "7_1"
    )]
    Surround7_1,
    #[value(name = "7.1.4", alias = "atmos-7.1.4", alias = "7_1_4")]
    Atmos7_1_4,
    #[value(name = "7.2.4", alias = "atmos-7.2.4", alias = "7_2_4")]
    Atmos7_2_4,
    #[value(name = "custom", alias = "layout-json")]
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RenderEngineArg {
    Auto,
    Direct,
    FftPartitioned,
}
