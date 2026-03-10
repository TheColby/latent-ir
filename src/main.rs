use anyhow::Result;
use clap::Parser;
use latent_ir::cli::Cli;
use tracing_subscriber::EnvFilter;

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    latent_ir::commands::dispatch(cli)
}
