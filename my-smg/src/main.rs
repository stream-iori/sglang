use my_smg::config::GatewayConfig;
use std::path::Path;
use std::process;

fn main() {
    let Some(config_path) = std::env::args().nth(1) else {
        eprintln!("usage: my-smg <config-path>");
        process::exit(2);
    };

    match GatewayConfig::from_file(Path::new(&config_path)) {
        Ok(config) => {
            println!(
                "configuration loaded: {} workers, policy: {:?}",
                config.workers.len(),
                config.policy
            );
        }
        Err(error) => {
            eprintln!("failed to start: {error}");
            process::exit(1);
        }
    }
}
