use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::cli::SampleArgs;
use crate::core::descriptors::DescriptorSet;
use crate::core::util;

pub fn run(args: SampleArgs) -> Result<()> {
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let mut out = Vec::with_capacity(args.count);

    for _ in 0..args.count {
        let mut d = DescriptorSet::default();
        d.time.t60 = rng.gen_range(0.4..16.0);
        d.time.predelay_ms = rng.gen_range(0.0..120.0);
        d.spectral.brightness = rng.gen_range(0.0..1.0);
        d.spectral.hf_damping = rng.gen_range(0.0..1.0);
        d.structural.diffusion = rng.gen_range(0.0..1.0);
        d.structural.early_density = rng.gen_range(0.0..1.0);
        d.structural.late_density = rng.gen_range(0.0..1.0);
        d.spatial.width = rng.gen_range(0.0..1.0);
        d.spatial.decorrelation = rng.gen_range(0.0..1.0);
        d.clamp();
        out.push(d);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        for (i, d) in out.iter().enumerate() {
            println!("{}", util::console::section(&format!("sample[{i}]")));
            println!(
                "{}",
                util::console::metric("t60_s", format!("{:.2}", d.time.t60))
            );
            println!(
                "{}",
                util::console::metric("predelay_ms", format!("{:.1}", d.time.predelay_ms))
            );
            println!(
                "{}",
                util::console::metric("diffusion", format!("{:.2}", d.structural.diffusion))
            );
        }
    }

    Ok(())
}
