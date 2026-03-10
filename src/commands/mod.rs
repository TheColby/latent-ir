use anyhow::Result;

use crate::cli::{Cli, Commands};

mod analyze;
mod generate;
mod morph;
mod preset;
mod render;
mod sample;
mod train_encoder;

pub fn dispatch(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Generate(args) => generate::run(args),
        Commands::Analyze(args) => analyze::run(args),
        Commands::Morph(args) => morph::run(args),
        Commands::Render(args) => render::run(args),
        Commands::Sample(args) => sample::run(args),
        Commands::Preset(args) => preset::run(args),
        Commands::TrainEncoder(args) => train_encoder::run(args),
    }
}
