//! A small HTTPS client shared by the providers: JSON and multipart POSTs, error bodies kept.

use std::io::Read;
use std::time::Duration;

use crate::AiError;

/// Largest response body read (a long transcript is a few MB of JSON).
const MAX_BODY: u64 = 64 * 1024 * 1024;

pub(crate) fn agent() -> ureq::Agent {
    let provider = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .unversioned_rustls_crypto_provider(provider)
                .build(),
        )
        .timeout_connect(Some(Duration::from_secs(30)))
        // long audio and long model answers take minutes
        .timeout_global(Some(Duration::from_secs(20 * 60)))
        // read 4xx/5xx bodies: they carry the provider's explanation
        .http_status_as_error(false)
        .build()
        .into()
}

/// The response status and body text.
pub(crate) struct Reply {
    pub status: u16,
    pub body: String,
}

fn read(provider: &'static str, r: Result<ureq::http::Response<ureq::Body>, ureq::Error>) -> Result<Reply, AiError> {
    let mut resp = r.map_err(|e| AiError::Network(e.to_string()))?;
    let status = resp.status().as_u16();
    let mut body = String::new();
    resp.body_mut()
        .as_reader()
        .take(MAX_BODY)
        .read_to_string(&mut body)
        .map_err(|e| AiError::BadResponse(provider, e.to_string()))?;
    Ok(Reply { status, body })
}

/// POST a JSON body with extra headers.
pub(crate) fn post_json(provider: &'static str, url: &str, headers: &[(&str, &str)], body: &serde_json::Value) -> Result<Reply, AiError> {
    let mut req = agent().post(url).header("content-type", "application/json");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    read(provider, req.send(body.to_string().as_bytes()))
}

/// One part of a multipart/form-data body.
pub(crate) enum Part<'a> {
    Text(&'a str, &'a str),
    File { name: &'a str, filename: &'a str, content_type: &'a str, data: &'a [u8] },
}

/// Build a multipart/form-data body; returns (content type header, body).
pub(crate) fn multipart(parts: &[Part]) -> (String, Vec<u8>) {
    let boundary = format!("montaj-{:016x}", seed());
    let mut b = Vec::new();
    for p in parts {
        b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        match p {
            Part::Text(name, value) => {
                b.extend_from_slice(format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
            }
            Part::File { name, filename, content_type, data } => {
                b.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n").as_bytes(),
                );
                b.extend_from_slice(data);
                b.extend_from_slice(b"\r\n");
            }
        }
    }
    b.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), b)
}

fn seed() -> u64 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    (t as u64) ^ ((t >> 64) as u64) ^ 0x9e37_79b9_7f4a_7c15
}

/// POST a multipart body with extra headers.
pub(crate) fn post_multipart(provider: &'static str, url: &str, headers: &[(&str, &str)], parts: &[Part]) -> Result<Reply, AiError> {
    let (ct, body) = multipart(parts);
    let mut req = agent().post(url).header("content-type", &ct);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    read(provider, req.send(&body[..]))
}

/// Turn a non-2xx reply into an error carrying the provider's message (never the request).
pub(crate) fn check(provider: &'static str, r: Reply) -> Result<String, AiError> {
    if (200..300).contains(&r.status) {
        return Ok(r.body);
    }
    let message = serde_json::from_str::<serde_json::Value>(&r.body)
        .ok()
        .and_then(|v| {
            v.pointer("/error/message").and_then(|m| m.as_str()).map(str::to_string).or_else(|| v.get("message").and_then(|m| m.as_str()).map(str::to_string))
        })
        .unwrap_or_else(|| r.body.chars().take(300).collect());
    Err(AiError::Http { provider, status: r.status, message })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_has_parts_and_closing_boundary() {
        let (ct, body) = multipart(&[Part::Text("model", "whisper-1"), Part::File { name: "file", filename: "a.wav", content_type: "audio/wav", data: b"RIFF" }]);
        let boundary = ct.split("boundary=").nth(1).unwrap();
        let s = String::from_utf8_lossy(&body);
        assert!(s.contains("name=\"model\"\r\n\r\nwhisper-1\r\n"));
        assert!(s.contains("filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\r\n"));
        assert!(s.ends_with(&format!("--{boundary}--\r\n")));
    }

    #[test]
    fn error_message_comes_from_the_body() {
        let e = check("OpenAI", Reply { status: 401, body: r#"{"error":{"message":"Incorrect API key provided"}}"#.into() }).unwrap_err();
        assert_eq!(e, AiError::Http { provider: "OpenAI", status: 401, message: "Incorrect API key provided".into() });
        assert_eq!(check("x", Reply { status: 200, body: "ok".into() }).unwrap(), "ok");
    }
}
