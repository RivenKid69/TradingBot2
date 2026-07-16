//! RivenQuant desktop shell (Tauri v2).
//!
//! The shell owns no business logic: it launches the Python backend as a bundled
//! sidecar, waits for it to report its loopback port and become reachable, then
//! opens the main webview directly against `http://127.0.0.1:<port>/` — so the UI
//! is byte-for-byte the existing MVP served by FastAPI. `window.RIVEN_API_BASE`
//! is injected before page scripts run so every `/api/*` call stays same-origin
//! regardless of the (dynamic) port. The sidecar is killed on app exit.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Holds the sidecar child and its handshake port for graceful shutdown.
#[derive(Default)]
struct SidecarProcess {
    child: Option<CommandChild>,
    port: Option<u16>,
}

struct SidecarHandle(Mutex<SidecarProcess>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarHandle(Mutex::new(SidecarProcess::default())))
        .setup(|app| {
            let handle = app.handle().clone();

            // Immediate splash from the bundled frontend dist (loading screen).
            let _ = WebviewWindowBuilder::new(&handle, "splash", WebviewUrl::App("index.html".into()))
                .title("RivenQuant")
                .inner_size(520.0, 340.0)
                .center()
                .decorations(false)
                .resizable(false)
                .always_on_top(false)
                .build();

            // Dev fast-path: connect to an already-running backend (no sidecar build).
            //   1) python desktop_backend.py --port 8002
            //   2) RIVEN_DEV_URL=http://127.0.0.1:8002 cargo tauri dev
            if let Ok(dev_url) = std::env::var("RIVEN_DEV_URL") {
                let dev_url = dev_url.trim().trim_end_matches('/').to_string();
                let h = handle.clone();
                tauri::async_runtime::spawn(async move {
                    if let Some((host, port)) = host_port_from_url(&dev_url) {
                        let _ = wait_for_host_port(&host, port, 90).await;
                    }
                    let target = format!("{dev_url}/");
                    let hh = h.clone();
                    let _ = h.run_on_main_thread(move || open_main_window(&hh, &target));
                });
                return Ok(());
            }

            // Launch the Python backend sidecar. NOTE: the runtime resolves the
            // name relative to the executable dir WITHOUT stripping path segments
            // (base_dir.join(name)), and the bundler places the binary next to the
            // exe as `riven-backend.exe`. So the runtime name is the bare basename,
            // even though `externalBin` in tauri.conf.json keeps the `binaries/` path.
            let sidecar = app
                .shell()
                .sidecar("riven-backend")
                .map_err(|e| format!("failed to resolve sidecar: {e}"))?;
            let (mut rx, child) = sidecar
                .spawn()
                .map_err(|e| format!("failed to spawn sidecar: {e}"))?;
            app.state::<SidecarHandle>()
                .0
                .lock()
                .expect("sidecar mutex poisoned")
                .child
                .replace(child);

            // Read the handshake on stdout, then open the real window once ready.
            tauri::async_runtime::spawn(async move {
                let mut port: Option<u16> = None;
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            for line in text.lines() {
                                if let Some(v) = line.trim().strip_prefix("RIVEN_PORT=") {
                                    if let Ok(p) = v.trim().parse::<u16>() {
                                        port = Some(p);
                                        handle
                                            .state::<SidecarHandle>()
                                            .0
                                            .lock()
                                            .expect("sidecar mutex poisoned")
                                            .port = Some(p);
                                    }
                                }
                            }
                            if let Some(p) = port {
                                if wait_for_host_port("127.0.0.1", p, 90).await {
                                    let h = handle.clone();
                                    let target = format!("http://127.0.0.1:{p}/");
                                    let _ = handle.run_on_main_thread(move || open_main_window(&h, &target));
                                } else {
                                    eprintln!("[shell] backend not reachable on port {p}");
                                }
                                break;
                            }
                        }
                        CommandEvent::Stderr(bytes) => {
                            eprint!("{}", String::from_utf8_lossy(&bytes));
                        }
                        CommandEvent::Error(err) => {
                            eprintln!("[shell] sidecar error: {err}");
                            break;
                        }
                        CommandEvent::Terminated(payload) => {
                            eprintln!("[shell] sidecar terminated: {:?}", payload.code);
                            break;
                        }
                        _ => {}
                    }
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the RivenQuant desktop application")
        .run(|app_handle, event| {
            if let RunEvent::ExitRequested { .. } = event {
                kill_sidecar(app_handle);
            }
        });
}

/// Poll a TCP connect to host:port until it accepts or we time out.
async fn wait_for_host_port(host: &str, port: u16, timeout_secs: u64) -> bool {
    let target = format!("{host}:{port}");
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    while Instant::now() < deadline {
        if tokio::net::TcpStream::connect(&target).await.is_ok() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

/// Extract (host, port) from a simple http URL for readiness polling.
fn host_port_from_url(url: &str) -> Option<(String, u16)> {
    let rest = url.strip_prefix("http://").or_else(|| url.strip_prefix("https://"))?;
    let authority = rest.split('/').next().unwrap_or(rest);
    let (host, port) = authority.split_once(':')?;
    Some((host.to_string(), port.parse().ok()?))
}

/// Create the main window pointed at the local backend and close the splash.
fn open_main_window(handle: &AppHandle, url: &str) {
    let parsed = match url.parse() {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[shell] bad backend url {url}: {e}");
            return;
        }
    };

    // Same-origin API base so /api/* calls hit the backend on any port.
    let init = "window.RIVEN_API_BASE='';";

    let result = WebviewWindowBuilder::new(handle, "main", WebviewUrl::External(parsed))
        .title("RivenQuant")
        .inner_size(1440.0, 900.0)
        .min_inner_size(1024.0, 680.0)
        .center()
        .initialization_script(init)
        .build();

    match result {
        Ok(_) => {
            if let Some(splash) = handle.get_webview_window("splash") {
                let _ = splash.close();
            }
        }
        Err(e) => eprintln!("[shell] failed to open main window: {e}"),
    }
}

/// Terminate the sidecar child if still running.
fn kill_sidecar(app_handle: &AppHandle) {
    if let Some(state) = app_handle.try_state::<SidecarHandle>() {
        let (port, child) = {
            let mut process = state.0.lock().expect("sidecar mutex poisoned");
            (process.port.take(), process.child.take())
        };
        if let Some(port) = port {
            request_backend_shutdown(port);
            // Uvicorn now exits itself after flushing CCEA. Give the PyInstaller
            // one-file child time to unwind before the force-kill fallback.
            std::thread::sleep(Duration::from_millis(750));
        }
        if let Some(child) = child {
            let _ = child.kill();
        }
    }
}

/// Ask the Python process to flush Agent/SQLite state before terminating it.
fn request_backend_shutdown(port: u16) {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    if let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(750)) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
        let request = format!(
            "POST /api/desktop/shutdown HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.write_all(request.as_bytes());
        let mut response = [0_u8; 256];
        let _ = stream.read(&mut response);
    }
}

#[cfg(test)]
mod tests {
    use super::host_port_from_url;

    #[test]
    fn parses_loopback_backend_url() {
        assert_eq!(
            host_port_from_url("http://127.0.0.1:8002/"),
            Some(("127.0.0.1".to_string(), 8002))
        );
    }

    #[test]
    fn rejects_url_without_explicit_port() {
        assert_eq!(host_port_from_url("https://example.com/path"), None);
    }
}
