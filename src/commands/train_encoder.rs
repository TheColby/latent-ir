use anyhow::Result;

use crate::cli::{TrainAudioEncoderArgs, TrainEncoderArgs, TrainEncoderMode, TrainTextEncoderArgs};
use crate::core::training::{
    train_audio_encoder, train_text_encoder, AudioTrainingConfig, TextTrainingConfig,
};
use crate::core::util;

pub fn run(args: TrainEncoderArgs) -> Result<()> {
    match args.mode {
        TrainEncoderMode::Text(cfg) => run_text(cfg),
        TrainEncoderMode::Audio(cfg) => run_audio(cfg),
    }
}

fn run_text(args: TrainTextEncoderArgs) -> Result<()> {
    let cfg = TextTrainingConfig {
        max_vocab: args.max_vocab,
        min_count: args.min_count,
        epochs: args.epochs,
        lr: args.lr,
        l2: args.l2,
    };

    let model = train_text_encoder(&args.dataset, &cfg)?;
    util::json::write_pretty_json(&args.output, &model)?;

    println!("trained text encoder model: {}", args.output.display());
    println!("embedding_dim: {}", model.embedding_dim);
    println!("vocab_size: {}", model.token_embeddings.len());
    Ok(())
}

fn run_audio(args: TrainAudioEncoderArgs) -> Result<()> {
    let cfg = AudioTrainingConfig {
        epochs: args.epochs,
        lr: args.lr,
        l2: args.l2,
    };

    let model = train_audio_encoder(&args.dataset, &cfg)?;
    util::json::write_pretty_json(&args.output, &model)?;

    println!("trained audio encoder model: {}", args.output.display());
    println!("feature_dim: {}", model.feature_names.len());
    println!("hidden_dim: {}", model.hidden_bias.len());
    Ok(())
}
