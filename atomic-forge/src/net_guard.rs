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
        || ip.is_multicast()
        || ip.is_broadcast()
        || o[0] == 0 // "this network" 0.0.0.0/8
        || o[0] >= 240 // reserved 240.0.0.0/4 (includes broadcast)
        || (o[0] == 100 && (o[1] & 0xC0) == 64) // CGNAT 100.64.0.0/10
        || (o[0] == 198 && (o[1] & 0xFE) == 18) // benchmarking 198.18.0.0/15
        || (o[0] == 192 && o[1] == 0 && o[2] == 0) // IETF protocol assignments 192.0.0.0/24
        || (o[0] == 192 && o[1] == 0 && o[2] == 2) // TEST-NET-1
        || (o[0] == 198 && o[1] == 51 && o[2] == 100) // TEST-NET-2
        || (o[0] == 203 && o[1] == 0 && o[2] == 113) // TEST-NET-3
}

fn v4_from_segments(hi: u16, lo: u16) -> Ipv4Addr {
    let [a, b] = hi.to_be_bytes();
    let [c, d] = lo.to_be_bytes();
    Ipv4Addr::new(a, b, c, d)
}

fn forbidden_v6(ip: Ipv6Addr) -> bool {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return forbidden_v4(v4);
    }
    let seg = ip.segments();
    if ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
        return true;
    }
    // IPv4-compatible ::a.b.c.d (deprecated); `::` and `::1` handled above.
    if seg[..6].iter().all(|&s| s == 0) {
        return forbidden_v4(v4_from_segments(seg[6], seg[7]));
    }
    // NAT64 well-known prefix 64:ff9b::/96 embeds the IPv4 in the low 32 bits.
    if seg[0] == 0x0064 && seg[1] == 0xff9b && seg[2..6].iter().all(|&s| s == 0) {
        return forbidden_v4(v4_from_segments(seg[6], seg[7]));
    }
    // NAT64 local-use prefix 64:ff9b:1::/48: the embedding is operator-defined, so block it all.
    if seg[0] == 0x0064 && seg[1] == 0xff9b && seg[2] == 0x0001 {
        return true;
    }
    // 6to4 2002::/16 embeds the IPv4 in bits 16..48.
    if seg[0] == 0x2002 {
        return forbidden_v4(v4_from_segments(seg[1], seg[2]));
    }
    // Teredo 2001::/32 embeds an obfuscated client IPv4; block the whole prefix.
    if seg[0] == 0x2001 && seg[1] == 0 {
        return true;
    }
    // Documentation 2001:db8::/32.
    if seg[0] == 0x2001 && seg[1] == 0x0db8 {
        return true;
    }
    (seg[0] & 0xFE00) == 0xFC00 // ULA fc00::/7
        || (seg[0] & 0xFFC0) == 0xFE80 // link-local fe80::/10
        || (seg[0] & 0xFFC0) == 0xFEC0 // deprecated site-local fec0::/10
}

/// True when `ip` is loopback, private, link-local, unspecified, multicast,
/// CGNAT, reserved/documentation/benchmarking ranges, IPv6 ULA, link-local or
/// site-local. IPv6 forms that embed an IPv4 address (IPv4-mapped, IPv4-compatible,
/// NAT64, 6to4) are judged by the embedded IPv4; Teredo and documentation
/// prefixes are rejected outright.
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

/// Apply the NetGuard client configuration to `builder`.
///
/// Proxies are disabled (`no_proxy`): a proxy picked up from `HTTP(S)_PROXY` /
/// `ALL_PROXY` would re-resolve the hostname itself and defeat the IP pinning.
fn configure_client(
    builder: reqwest::ClientBuilder,
    user_agent: &str,
    timeout: Duration,
) -> reqwest::ClientBuilder {
    builder
        .no_proxy()
        .user_agent(user_agent)
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
}

/// Guard configuration.
///
/// Every client built by [`NetGuard::get`] ignores proxy environment variables
/// so that connections really go to the validated, pinned addresses.
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
            let mut builder = configure_client(reqwest::Client::builder(), user_agent, timeout);
            if let Some(host) = current.host_str()
                && host.parse::<IpAddr>().is_err()
                && !host.starts_with('[')
            {
                builder = builder.resolve_to_addrs(host, &addrs);
            }
            let client = builder
                .build()
                .map_err(|e| format!("Failed to build HTTP client: {e}"))?;
            let response = client
                .get(current.clone())
                .header("Accept", accept)
                .send()
                .await
                .map_err(|e| format!("Request failed: {}", e.without_url()))?;
            // Only genuine redirects are followed; other 3xx (300, 304, ...) go to the caller.
            if matches!(response.status().as_u16(), 301 | 302 | 303 | 307 | 308) {
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
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254",
            "0.0.0.0",
            "224.0.0.1",
            "255.255.255.255",
            "100.64.0.1",
            "100.127.255.255",
            "::1",
            "::",
            "ff02::1",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "::ffff:169.254.169.254",
            "::ffff:192.168.0.1",
        ] {
            assert!(is_forbidden_ip(ip(s)), "{s} should be forbidden");
        }
    }

    #[test]
    fn allows_public_addresses() {
        for s in [
            "8.8.8.8",
            "1.1.1.1",
            "172.15.0.1",
            "172.32.0.1",
            "100.63.255.255",
            "100.128.0.1",
            "2606:4700:4700::1111",
            "::ffff:8.8.8.8",
        ] {
            assert!(!is_forbidden_ip(ip(s)), "{s} should be allowed");
        }
    }

    #[test]
    fn classifies_extended_forbidden_ranges() {
        for s in [
            // NAT64 and local-use NAT64
            "64:ff9b::7f00:1",
            "64:ff9b::a9fe:a9fe",
            "64:ff9b::10.0.0.1",
            "64:ff9b:1::1",
            "64:ff9b:1:ffff::1",
            // 6to4
            "2002:7f00:1::1",
            "2002:a9fe:a9fe::1",
            "2002:c0a8:101::",
            "2002:0a00:0001::5",
            // IPv4-compatible
            "::127.0.0.1",
            "::10.0.0.1",
            "::169.254.169.254",
            "::0.0.0.5",
            // site-local, Teredo, documentation
            "fec0::1",
            "feff::1",
            "2001::1",
            "2001:0:4136:e378:8000:63bf:3fff:fdd2",
            "2001:db8::1",
            // IPv4 ranges
            "0.1.2.3",
            "0.255.255.255",
            "198.18.0.1",
            "198.19.255.255",
            "240.0.0.1",
            "250.1.1.1",
            "192.0.0.1",
            "192.0.2.5",
            "198.51.100.7",
            "203.0.113.9",
        ] {
            assert!(is_forbidden_ip(ip(s)), "{s} should be forbidden");
        }
    }

    #[test]
    fn extended_ranges_do_not_over_block() {
        for s in [
            "8.8.8.8",
            "2606:4700::1111",
            "64:ff9b::808:808",
            "2002:808:808::1",
            "::8.8.8.8",
            "198.17.255.255",
            "198.20.0.1",
            "192.0.1.1",
            "192.0.3.1",
            "198.51.101.1",
            "203.0.112.1",
            "223.255.255.255",
            "2001:4860:4860::8888",
            "2001:1::1",
            "fe00::1",
        ] {
            assert!(!is_forbidden_ip(ip(s)), "{s} should be allowed");
        }
    }

    #[tokio::test]
    async fn configured_client_ignores_proxies() {
        let proxy = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string("proxy"))
            .mount(&proxy)
            .await;
        let target = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string("direct"))
            .mount(&target)
            .await;
        let base = reqwest::Client::builder().proxy(reqwest::Proxy::all(proxy.uri()).unwrap());
        let client = configure_client(base, "t", Duration::from_secs(5))
            .build()
            .unwrap();
        let body = client
            .get(target.uri())
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert_eq!(body, "direct");
        assert!(proxy.received_requests().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn non_redirect_3xx_is_returned_to_caller() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(300))
            .mount(&server)
            .await;
        let guard = NetGuard::new(true);
        let resp = guard
            .get(&server.uri(), "t", "*/*", Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 300);

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(304).insert_header("Location", "http://169.254.169.254/"),
            )
            .mount(&server)
            .await;
        let resp = guard
            .get(&server.uri(), "t", "*/*", Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 304);
    }

    #[tokio::test]
    async fn validate_rejects_loopback_and_bad_schemes() {
        let guard = NetGuard::new(false);
        let err = guard
            .validate(&"http://127.0.0.1:80/".parse().unwrap())
            .await
            .unwrap_err();
        assert!(err.contains("non-public"));
        assert!(
            guard
                .validate(&"http://[::1]/".parse().unwrap())
                .await
                .is_err()
        );
        assert!(
            guard
                .validate(&"ftp://example.com/".parse().unwrap())
                .await
                .is_err()
        );
        assert!(
            NetGuard::new(true)
                .validate(&"http://127.0.0.1:80/".parse().unwrap())
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn redirect_to_private_host_is_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(
                ResponseTemplate::new(302).insert_header("Location", "http://127.0.0.2:9/x"),
            )
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
        assert!(
            err.contains("127.0.0.2") && err.contains("non-public"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn redirects_are_followed_and_limited() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(302).insert_header("Location", "/again"))
            .mount(&server)
            .await;
        let guard = NetGuard::new(true);
        let err = guard
            .get(&server.uri(), "t", "*/*", Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(err.contains("Too many redirects"));
    }
}
