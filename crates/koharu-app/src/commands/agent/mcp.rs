//! The Agent's tools over MCP (Streamable HTTP, JSON responses) for agents
//! outside the app, such as Claude Code. They act on the open project through
//! the same host as the in-app Agent: the same project lock, the same undo
//! history. The server listens on the loopback interface only, and refuses
//! requests a web page could send (a foreign `Origin`).

use anyhow::{Context as _, Result, bail};
use koharu_agent::{Control, Host as _, ToolCall};
use serde_json::{Value, json};
use tauri::AppHandle;
use tauri_runtime_cef::CefRuntime;
use tokio::{
    io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader},
    net::{TcpListener, TcpStream},
};

use super::host::KoharuHost;

/// Loopback address and port of the server; `claude mcp add --transport http
/// koharu http://127.0.0.1:4100/mcp` connects to it.
const ADDRESS: &str = "127.0.0.1:4100";
const PROTOCOL_VERSION: &str = "2025-06-18";
const MAX_BODY: usize = 16 * 1024 * 1024;

/// Serves until the app exits. A second Koharu on the same machine finds the
/// port taken and runs without the server.
pub(crate) async fn serve(handle: AppHandle<CefRuntime>) {
    let listener = match TcpListener::bind(ADDRESS).await {
        Ok(listener) => listener,
        Err(error) => {
            tracing::warn!(%error, "the MCP server could not listen on {ADDRESS}");
            return;
        }
    };
    tracing::info!("MCP server listening on http://{ADDRESS}/mcp");
    let host = KoharuHost::new(handle);
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let host = host.clone();
        drop(tauri::async_runtime::spawn(async move {
            if let Err(error) = connection(stream, &host).await {
                tracing::debug!(%error, "MCP connection ended");
            }
        }));
    }
}

struct Request {
    method: String,
    path: String,
    origin: Option<String>,
    body: Vec<u8>,
    keep_alive: bool,
}

async fn connection(stream: TcpStream, host: &KoharuHost) -> Result<()> {
    let mut stream = BufReader::new(stream);
    loop {
        let Some(request) = read_request(&mut stream).await? else {
            return Ok(());
        };
        let keep_alive = request.keep_alive;
        let (status, body) = respond(request, host).await;
        let mut head = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n",
            body.as_ref().map_or(0, Vec::len)
        );
        if body.is_some() {
            head.push_str("Content-Type: application/json\r\n");
        }
        head.push_str(if keep_alive {
            "Connection: keep-alive\r\n\r\n"
        } else {
            "Connection: close\r\n\r\n"
        });
        let writer = stream.get_mut();
        writer.write_all(head.as_bytes()).await?;
        if let Some(body) = body {
            writer.write_all(&body).await?;
        }
        writer.flush().await?;
        if !keep_alive {
            return Ok(());
        }
    }
}

async fn read_request(stream: &mut BufReader<TcpStream>) -> Result<Option<Request>> {
    let mut line = String::new();
    if stream.read_line(&mut line).await? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().context("empty request line")?.to_owned();
    let path = parts.next().context("request line has no path")?.to_owned();
    let version = parts.next().unwrap_or("HTTP/1.0").to_owned();
    let (mut length, mut origin, mut connection) = (0usize, None, None);
    loop {
        line.clear();
        if stream.read_line(&mut line).await? == 0 {
            bail!("connection closed inside the request headers");
        }
        let header = line.trim_end();
        if header.is_empty() {
            break;
        }
        let Some((name, value)) = header.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => length = value.parse().context("invalid Content-Length")?,
            "origin" => origin = Some(value.to_owned()),
            "connection" => connection = Some(value.to_ascii_lowercase()),
            _ => {}
        }
    }
    if length > MAX_BODY {
        bail!("request body is too large");
    }
    let mut body = vec![0; length];
    stream.read_exact(&mut body).await?;
    let keep_alive = match connection.as_deref() {
        Some("close") => false,
        Some("keep-alive") => true,
        _ => version == "HTTP/1.1",
    };
    Ok(Some(Request {
        method,
        path,
        origin,
        body,
        keep_alive,
    }))
}

async fn respond(request: Request, host: &KoharuHost) -> (&'static str, Option<Vec<u8>>) {
    // A browser page could post to a loopback port; only local origins pass.
    if request.origin.as_deref().is_some_and(|origin| {
        ![
            "http://127.0.0.1",
            "http://localhost",
            "https://127.0.0.1",
            "https://localhost",
        ]
        .iter()
        .any(|local| {
            origin == *local
                || origin
                    .strip_prefix(local)
                    .is_some_and(|rest| rest.starts_with(':'))
        })
    }) {
        return ("403 Forbidden", None);
    }
    if request.path.split('?').next() != Some("/mcp") {
        return ("404 Not Found", None);
    }
    if request.method != "POST" {
        return ("405 Method Not Allowed", None);
    }
    let message = match serde_json::from_slice::<Value>(&request.body) {
        Ok(message) => message,
        Err(error) => {
            return (
                "400 Bad Request",
                Some(error_response(&Value::Null, -32700, &error.to_string())),
            );
        }
    };
    let Some(id) = message.get("id").cloned() else {
        // Notifications and responses from the client need no answer.
        return ("202 Accepted", None);
    };
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(PROTOCOL_VERSION),
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": "koharu", "version": env!("CARGO_PKG_VERSION") },
            "instructions": "Tools act on the project open in Koharu, with the same \
                undo history as the app. Start with inspect_project; element and page \
                IDs come from it.",
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({
            "tools": host
                .tools()
                .into_iter()
                .map(|tool| json!({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.parameters,
                }))
                .collect::<Vec<_>>(),
        })),
        "tools/call" => Ok(call(host, &id, &params).await),
        _ => Err((-32601, format!("method {method} is not supported"))),
    };
    let body = match result {
        Ok(result) => serde_json::to_vec(&json!({ "jsonrpc": "2.0", "id": id, "result": result }))
            .unwrap_or_default(),
        Err((code, message)) => error_response(&id, code, &message),
    };
    ("200 OK", Some(body))
}

/// One tool call; a failed call is a tool result marked as an error, so the
/// calling agent reads the reason and can correct its arguments.
async fn call(host: &KoharuHost, id: &Value, params: &Value) -> Value {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let call = ToolCall {
        call_id: id.to_string(),
        name,
        arguments: arguments.to_string(),
    };
    match host.invoke(call, &Control::default()).await {
        Ok(invocation) => {
            let mut content = vec![json!({
                "type": "text",
                "text": serde_json::to_string_pretty(&invocation.value).unwrap_or_default(),
            })];
            for image in invocation.images {
                if let Some((mime, data)) = image
                    .data_url
                    .strip_prefix("data:")
                    .and_then(|rest| rest.split_once(";base64,"))
                {
                    content.push(json!({ "type": "text", "text": image.label }));
                    content.push(json!({ "type": "image", "data": data, "mimeType": mime }));
                }
            }
            json!({ "content": content, "isError": false })
        }
        Err(error) => json!({
            "content": [{ "type": "text", "text": format!("{error:#}") }],
            "isError": true,
        }),
    }
}

fn error_response(id: &Value, code: i64, message: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    }))
    .unwrap_or_default()
}
