use anyhow::{anyhow, Result};

use crate::cli::{EvalArgs, EvalAudioArgs, EvalCheckArgs, EvalMode, EvalTextArgs};
use crate::core::eval::{check_eval, evaluate_audio, evaluate_text, BaselineReport};
use crate::core::util;

pub fn run(args: EvalArgs) -> Result<()> {
    match args.mode {
        EvalMode::Text(cfg) => run_text(cfg),
        EvalMode::Audio(cfg) => run_audio(cfg),
        EvalMode::Check(cfg) => run_check(cfg),
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

fn run_check(args: EvalCheckArgs) -> Result<()> {
    let report: BaselineReport = serde_json::from_str(&std::fs::read_to_string(&args.report)?)?;
    let baseline: BaselineReport = serde_json::from_str(&std::fs::read_to_string(&args.baseline)?)?;

    let result = check_eval(&report, &baseline, args.max_regression);
    if result.passed {
        println!("{}", util::console::success("eval check passed"));
        return Ok(());
    }

    for r in result.regressions {
        eprintln!("{}", util::console::error(&format!("regression: {r}")));
    }
    Err(anyhow!("eval check failed"))
}
