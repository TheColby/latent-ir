use anyhow::{Context, Result};

use crate::cli::PresetArgs;
use crate::core::presets;

pub fn run(args: PresetArgs) -> Result<()> {
    match args.name {
        Some(name) => {
            let preset = presets::resolve_preset(&name)
                .with_context(|| format!("unknown preset '{name}'"))?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&preset)?);
            } else {
                println!("preset: {name}");
                println!("{}", serde_json::to_string_pretty(&preset)?);
            }
        }
        None => {
            let names = presets::preset_names();
            if args.json {
                println!("{}", serde_json::to_string_pretty(&names)?);
            } else {
                for n in names {
                    println!("{n}");
                }
            }
        }
    }

    Ok(())
}
