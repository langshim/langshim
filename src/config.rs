use dotenvy::dotenv;
use std::env;
use std::net::IpAddr;
use std::path::PathBuf;

#[derive(Clone)]
pub struct Config {
    pub data_dir: PathBuf,
    pub api_key: String,
    pub host: IpAddr,
    pub port: u16,
    pub live_exchange: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConfigOverrides {
    pub data_dir: Option<PathBuf>,
    pub api_key: Option<String>,
    pub host: Option<IpAddr>,
    pub port: Option<u16>,
    pub live_exchange: Option<bool>,
}

impl Config {
    pub fn from_sources(overrides: ConfigOverrides) -> Self {
        dotenv().ok();

        Self {
            data_dir: langshim_data_dir(overrides.data_dir),
            api_key: overrides
                .api_key
                .or_else(|| env::var("LANGSHIM_API_KEY").ok())
                .unwrap_or_else(|| "secret".to_string()),
            host: overrides
                .host
                .or_else(langshim_host_from_env)
                .unwrap_or(IpAddr::from([127, 0, 0, 1])),
            port: overrides
                .port
                .or_else(langshim_port_from_env)
                .unwrap_or(3000),
            live_exchange: overrides.live_exchange.unwrap_or(false),
        }
    }
}

pub fn langshim_data_dir(override_path: Option<PathBuf>) -> PathBuf {
    if let Some(path) = override_path {
        return path;
    }

    if let Some(value) = env::var_os("LANGSHIM_HOME") {
        return PathBuf::from(value);
    }

    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home).join(".langshim");
    }

    env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".langshim")
}

fn langshim_port_from_env() -> Option<u16> {
    env::var("LANGSHIM_PORT")
        .ok()
        .map(|value| parse_port(&value, "LANGSHIM_PORT"))
}

fn langshim_host_from_env() -> Option<IpAddr> {
    env::var("LANGSHIM_HOST")
        .ok()
        .map(|value| parse_host(&value, "LANGSHIM_HOST"))
}

pub fn parse_port(value: &str, source: &str) -> u16 {
    value
        .parse::<u16>()
        .unwrap_or_else(|_| panic!("Invalid port in {}: {}", source, value))
}

pub fn parse_host(value: &str, source: &str) -> IpAddr {
    value
        .parse::<IpAddr>()
        .unwrap_or_else(|_| panic!("Invalid host in {}: {}", source, value))
}

#[cfg(test)]
mod tests {
    use super::{Config, ConfigOverrides, langshim_data_dir, parse_host, parse_port};
    use std::net::IpAddr;
    use std::path::PathBuf;

    #[test]
    fn command_line_overrides_environment() {
        unsafe {
            std::env::set_var("LANGSHIM_HOME", "/tmp/from-env");
            std::env::set_var("LANGSHIM_API_KEY", "env-secret");
            std::env::set_var("LANGSHIM_HOST", "0.0.0.0");
            std::env::set_var("LANGSHIM_PORT", "9000");
        }

        let config = Config::from_sources(ConfigOverrides {
            data_dir: Some(PathBuf::from("/tmp/from-cli")),
            api_key: Some("cli-secret".to_string()),
            host: Some(IpAddr::from([127, 0, 0, 2])),
            port: Some(7000),
            live_exchange: Some(true),
        });

        assert_eq!(config.data_dir, PathBuf::from("/tmp/from-cli"));
        assert_eq!(config.api_key, "cli-secret");
        assert_eq!(config.host, IpAddr::from([127, 0, 0, 2]));
        assert_eq!(config.port, 7000);
        assert!(config.live_exchange);

        unsafe {
            std::env::remove_var("LANGSHIM_HOME");
            std::env::remove_var("LANGSHIM_API_KEY");
            std::env::remove_var("LANGSHIM_HOST");
            std::env::remove_var("LANGSHIM_PORT");
        }
    }

    #[test]
    fn data_dir_uses_environment_when_cli_override_missing() {
        unsafe {
            std::env::set_var("LANGSHIM_HOME", "/tmp/from-env");
        }

        assert_eq!(langshim_data_dir(None), PathBuf::from("/tmp/from-env"));

        unsafe {
            std::env::remove_var("LANGSHIM_HOME");
        }
    }

    #[test]
    fn parse_port_accepts_valid_port() {
        assert_eq!(parse_port("3000", "test"), 3000);
    }

    #[test]
    fn parse_host_accepts_valid_ip_address() {
        assert_eq!(parse_host("0.0.0.0", "test"), IpAddr::from([0, 0, 0, 0]));
    }
}
