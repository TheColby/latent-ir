use anyhow::Result;

use crate::cli::{EvalArgs, EvalAudioArgs, EvalMode, EvalTextArgs};
use crate::core::eval::{evaluate_audio, evaluate_text};
use crate::core::util;

pub fn run(args: EvalArgs) -> Result<()> {
    match args.mode {
        EvalMode::Text(cfg) => run_text(cfg),
        EvalMode::Audio(cfg) => run_audio(cfg),
    }
}

fn run_text(args: EvalTextArgs) -> Result<()> {
    let report = evaluate_text(&args.dataset, &args.model, args.sample_rate, args.seed)?;
    util::json::write_pretty_json(&args.output, &report)?;

    println!(
        "{}",
        util::console::info("wrote eval baseline", args.output.display().to_string())
    );
    println!("{}", util::console::metric("samples", report.sample_count));
    println!(
        "{}",
        util::console::metric(
            "descriptor_mae",
            format!("{:.6}", report.descriptor_metrics.mae)
        )
    );
    println!(
        "{}",
        util::console::metric(
            "analysis_mae",
            format!("{:.6}", report.analysis_metrics.mae)
        )
    );
    Ok(())
}

fn run_audio(args: EvalAudioArgs) -> Result<()> {
    let report = evaluate_audio(&args.dataset, &args.model, args.sample_rate, args.seed)?;
    util::json::write_pretty_json(&args.output, &report)?;

    println!(
        "{}",
        util::console::info("wrote eval baseline", args.output.display().to_string())
    );
    println!("{}", util::console::metric("samples", report.sample_count));
    println!(
        "{}",
        util::console::metric(
            "descriptor_mae",
            format!("{:.6}", report.descriptor_metrics.mae)
        )
    );
    println!(
        "{}",
        util::console::metric(
            "analysis_mae",
            format!("{:.6}", report.analysis_metrics.mae)
        )
    );
    Ok(())
}
