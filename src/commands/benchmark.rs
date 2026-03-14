use anyhow::{anyhow, Result};

use crate::cli::{
    BenchmarkArgs, BenchmarkCheckArgs, BenchmarkMode, BenchmarkRunArgs, BenchmarkTrendArgs,
};
use crate::core::benchmark::{
    build_trend_report, check_benchmark, render_trend_markdown, run_benchmark, BenchmarkReport,
    BenchmarkRunConfig,
};
use crate::core::util;

pub fn run(args: BenchmarkArgs) -> Result<()> {
    match args.mode {
        BenchmarkMode::Run(cfg) => run_suite(cfg),
        BenchmarkMode::Check(cfg) => run_check(cfg),
        BenchmarkMode::Trend(cfg) => run_trend(cfg),
    }
}

fn run_suite(args: BenchmarkRunArgs) -> Result<()> {
    let cfg = BenchmarkRunConfig {
        dataset_path: args.dataset,
        sample_rate: args.sample_rate,
        seed: args.seed,
        repeats: args.repeats,
        text_model: args.text_model,
        audio_model: args.audio_model,
    };
    let report = run_benchmark(&cfg)?;
    util::json::write_pretty_json(&args.output, &report)?;

    println!(
        "{}",
        util::console::info("wrote benchmark report", args.output.display().to_string())
    );
    println!("{}", util::console::metric("samples", report.sample_count));
    println!(
        "{}",
        util::console::metric(
            "objective_score",
            format!("{:.6}", report.summary.objective_score)
        )
    );
    println!(
        "{}",
        util::console::metric("total_score", format!("{:.6}", report.summary.total_score))
    );
    Ok(())
}

fn run_check(args: BenchmarkCheckArgs) -> Result<()> {
    let report: BenchmarkReport = serde_json::from_str(&std::fs::read_to_string(&args.report)?)?;
    let baseline: BenchmarkReport =
        serde_json::from_str(&std::fs::read_to_string(&args.baseline)?)?;

    let result = check_benchmark(&report, &baseline, args.max_regression);
    if result.passed {
        println!("{}", util::console::success("benchmark check passed"));
        return Ok(());
    }

    for r in result.regressions {
        eprintln!("{}", util::console::error(&format!("regression: {r}")));
    }
    Err(anyhow!("benchmark check failed"))
}

fn run_trend(args: BenchmarkTrendArgs) -> Result<()> {
    anyhow::ensure!(
        args.reports.len() >= 2,
        "benchmark trend requires at least 2 report paths"
    );
    let mut reports = Vec::with_capacity(args.reports.len());
    for path in &args.reports {
        let report: BenchmarkReport = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        reports.push((path.display().to_string(), report));
    }
    let trend = build_trend_report(&reports);
    let markdown = render_trend_markdown(&trend);
    std::fs::write(&args.output, markdown)?;

    let json_out = args.output.with_extension("json");
    util::json::write_pretty_json(&json_out, &trend)?;

    println!(
        "{}",
        util::console::info("trend_markdown", args.output.display().to_string())
    );
    println!(
        "{}",
        util::console::info("trend_json", json_out.display().to_string())
    );
    println!(
        "{}",
        util::console::metric("points", trend.summary.point_count)
    );
    println!(
        "{}",
        util::console::metric(
            "delta_from_first_pct",
            format!("{:.2}", trend.summary.relative_delta_from_first * 100.0)
        )
    );
    Ok(())
}
