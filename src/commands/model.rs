use anyhow::{anyhow, Result};

use crate::cli::{ModelArgs, ModelMode, ModelValidateArgs};
use crate::core::model::{load_manifest, validate_manifest, RuntimeCapabilities};

pub fn run(args: ModelArgs) -> Result<()> {
    match args.mode {
        ModelMode::Validate(v) => run_validate(v),
    }
}

fn run_validate(args: ModelValidateArgs) -> Result<()> {
    let manifest = load_manifest(&args.manifest)?;
    let runtime = RuntimeCapabilities::current();
    validate_manifest(&manifest, &runtime).map_err(|e| anyhow!("validation failed: {e}"))?;

    println!("manifest valid: {}", args.manifest.display());
    println!("model: {}", manifest.name);
    Ok(())
}
