//! SSRF guard for tools that fetch caller-supplied URLs.
//!
//! [`NetGuard`] resolves a URL's host, rejects non-public addresses (unless
//! explicitly allowed), pins the connection to the validated addresses so DNS
//! rebinding cannot swap them, and follows redirects manually, re-validating
//! every hop.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

/// Maximum number of redirects followed by [`NetGuard::get`].
pub const MAX_REDIRECTS: usize = 5;

/// True for an IPv4 address that is not publicly routable.
fn forbidden_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_multicast()
        || ip.is_broadcast()
        || (o[0] == 100 && (o[1] & 0xC0) == 64) // CGNAT 100.64.0.0/10
}

fn forbidden_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return forbidden_v4(v4);
    }
    let seg = ip.segments();
    ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (seg[0] & 0xFE00) == 0xFC00 // ULA fc00::/7
        || (seg[0] & 0xFFC0) == 0xFE80 // link-local fe80::/10
}

/// True when `ip` is loopback, private, link-local, unspecified, multicast,
/// CGNAT, IPv6 ULA or IPv6 link-local. IPv4-mapped IPv6 is treated as IPv4.
///
/// ```
/// use atomic_forge::net_guard::is_forbidden_ip;
/// assert!(is_forbidden_ip("169.254.169.254".parse().unwrap()));
/// assert!(is_forbidden_ip("::ffff:10.0.0.1".parse().unwrap()));
/// assert!(!is_forbidden_ip("8.8.8.8".parse().unwrap()));
/// ```
pub fn is_forbidden_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => forbidden_v4(v4),
        IpAddr::V6(v6) => forbidden_v6(v6),
    }
}

/// Guard configuration.
#[derive(Clone, Debug, Default)]
pub struct NetGuard {
    /// Skip the address checks (loopback, private ranges, ...).
    pub allow_private_hosts: bool,
    #[cfg(test)]
    pub(crate) exempt: Vec<IpAddr>,
}

impl NetGuard {
    pub fn new(allow_private_hosts: bool) -> Self {
        Self {
            allow_private_hosts,
            #[cfg(test)]
            exempt: Vec::new(),
        }
    }

    fn permitted(&self, ip: IpAddr) -> bool {
        if self.allow_private_hosts || !is_forbidden_ip(ip) {
            return true;
        }
        #[cfg(test)]
        if self.exempt.contains(&ip) {
            return true;
        }
        false
    }

    /// Resolve the URL's host and return the validated socket addresses.
    /// Fails if the scheme is not http(s) or any resolved address is forbidden.
    pub async fn validate(&self, url: &url::Url) -> Result<Vec<SocketAddr>, String> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(format!("Unsupported URL scheme '{}'.", url.scheme()));
        }
        let host = url.host_str().ok_or("URL has no host.")?;
        let port = url.port_or_known_default().ok_or("URL has no port.")?;
        let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host.trim_matches(['[', ']']), port))
            .await
            .map_err(|e| format!("Failed to resolve host '{host}': {e}"))?
            .collect();
        if addrs.is_empty() {
            return Err(format!("Host '{host}' did not resolve to any address."));
        }
        if let Some(bad) = addrs.iter().find(|a| !self.permitted(a.ip())) {
            return Err(format!(
                "Refusing to fetch '{host}': it resolves to non-public address {}.",
                bad.ip()
            ));
        }
        Ok(addrs)
    }

    /// GET `url`, validating the host before the request and again for each of up
    /// to [`MAX_REDIRECTS`] redirects. Connections are pinned to the validated IPs.
    pub async fn get(
        &self,
        url: &str,
        user_agent: &str,
        accept: &str,
        timeout: Duration,
    ) -> Result<reqwest::Response, String> {
        let mut current = url::Url::parse(url).map_err(|e| format!("Invalid URL '{url}': {e}"))?;
        for _ in 0..=MAX_REDIRECTS {
            let addrs = self.validate(&current).await?;
            let mut builder = reqwest::Client::builder()
                .user_agent(user_agent)
                .timeout(timeout)
                .redirect(reqwest::redirect::Policy::none());
            if let Some(host) = current.host_str()
                && host.parse::<IpAddr>().is_err()
                && !host.starts_with('[')
            {
                builder = builder.resolve_to_addrs(host, &addrs);
            }
            let client = builder.build().map_err(|e| format!("Failed to build HTTP client: {e}"))?;
            let response = client
                .get(current.clone())
                .header("Accept", accept)
                .send()
                .await
                .map_err(|e| format!("Request failed: {}", e.without_url()))?;
            if response.status().is_redirection() {
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or("Redirect response is missing a Location header.")?;
                current = current
                    .join(location)
                    .map_err(|e| format!("Invalid redirect Location '{location}': {e}"))?;
                continue;
            }
            return Ok(response);
        }
        Err(format!("Too many redirects (more than {MAX_REDIRECTS})."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn classifies_forbidden_addresses() {
        for s in [
            "127.0.0.1", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.1.1",
            "169.254.169.254", "0.0.0.0", "224.0.0.1", "255.255.255.255", "100.64.0.1",
            "100.127.255.255", "::1", "::", "ff02::1", "fc00::1", "fd12:3456::1", "fe80::1",
            "::ffff:127.0.0.1", "::ffff:169.254.169.254", "::ffff:192.168.0.1",
        ] {
            assert!(is_forbidden_ip(ip(s)), "{s} should be forbidden");
        }
    }

    #[test]
    fn allows_public_addresses() {
        for s in [
            "8.8.8.8", "1.1.1.1", "172.15.0.1", "172.32.0.1", "100.63.255.255",
            "100.128.0.1", "2606:4700:4700::1111", "::ffff:8.8.8.8",
        ] {
            assert!(!is_forbidden_ip(ip(s)), "{s} should be allowed");
        }
    }

    #[tokio::test]
    async fn validate_rejects_loopback_and_bad_schemes() {
        let guard = NetGuard::new(false);
        let err = guard.validate(&"http://127.0.0.1:80/".parse().unwrap()).await.unwrap_err();
        assert!(err.contains("non-public"));
        assert!(guard.validate(&"http://[::1]/".parse().unwrap()).await.is_err());
        assert!(guard.validate(&"ftp://example.com/".parse().unwrap()).await.is_err());
        assert!(NetGuard::new(true).validate(&"http://127.0.0.1:80/".parse().unwrap()).await.is_ok());
    }

    #[tokio::test]
    async fn redirect_to_private_host_is_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(302).insert_header("Location", "http://127.0.0.2:9/x"))
            .mount(&server)
            .await;
        // The first hop (127.0.0.1) is exempt; the redirect target must be re-validated.
        let guard = NetGuard {
            allow_private_hosts: false,
            exempt: vec![ip("127.0.0.1")],
        };
        let err = guard
            .get(&server.uri(), "t", "*/*", Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(err.contains("127.0.0.2") && err.contains("non-public"), "{err}");
    }

    #[tokio::test]
    async fn redirects_are_followed_and_limited() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(302).insert_header("Location", "/again"))
            .mount(&server)
            .await;
        let guard = NetGuard::new(true);
        let err = guard.get(&server.uri(), "t", "*/*", Duration::from_secs(5)).await.unwrap_err();
        assert!(err.contains("Too many redirects"));
    }
}
