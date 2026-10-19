#!/usr/bin/env python3
"""
health_check.py -- mandatory pre-flight check for the news agent.

Verifies, before the agent does any work:

  1. the Ollama daemon answers  (preferred: GET /api/version, then the
     `ollama list` / `ollama ps` CLIs as a fallback), and
  2. the Ollama model the agent is about to use is actually pulled.

It prints one readable PASS/FAIL line per check plus an actionable hint, and it
never raises a traceback: any unexpected error is reported as a failed check.

Standalone:
    python3 health_check.py
    python3 health_check.py --model qwen2.5:3b
    python3 health_check.py --host 127.0.0.1:11434 --quiet

As a library (used by news_agent.py):
    import health_check
    ok = health_check.run(model="llama3")

Exit codes: 0 = every check passed, 1 = at least one check failed,
2 = bad command line.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("NEWS_AGENT_MODEL", "llama3")
DEFAULT_TIMEOUT = 5.0

PASS = "PASS"
FAIL = "FAIL"


# ---------------------------------------------------------------------------
#  small helpers
# ---------------------------------------------------------------------------

def normalize_host(host: str | None) -> str:
    """Accept '127.0.0.1:11434', 'http://box:11434/' or an empty string."""
    host = (host or "").strip() or DEFAULT_HOST
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/")


def _color(code: str, text: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def _http_json(url: str, timeout: float) -> dict:
    """GET a JSON endpoint with the standard library only (no deps needed)."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
    data = json.loads(body or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"unexpected payload from {url}")
    return data


def _run_cli(*args: str, timeout: float = 10.0) -> Optional[tuple[int, str, str]]:
    """Run `ollama <args>`. Returns (returncode, stdout, stderr) or None."""
    binary = shutil.which("ollama")
    if not binary:
        return None
    try:
        proc = subprocess.run(
            [binary, *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _cli_model_names(output: str) -> list[str]:
    """Parse the NAME column out of `ollama list` output."""
    names: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("NAME"):
            continue
        names.append(line.split()[0])
    return names


def model_matches(installed: Sequence[str], wanted: str) -> bool:
    """'llama3' matches 'llama3:latest'; 'qwen2.5:3b' needs that exact tag."""
    wanted = (wanted or "").strip()
    if not wanted:
        return False
    base = wanted.split(":")[0]
    for tag in installed:
        tag = tag.strip()
        if tag == wanted or tag == f"{wanted}:latest":
            return True
        if ":" not in wanted and tag.split(":")[0] == base:
            return True
    return False


# ---------------------------------------------------------------------------
#  checks
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""
    method: str = ""

    @property
    def status(self) -> str:
        return PASS if self.ok else FAIL


def check_ollama_daemon(host: str, timeout: float = DEFAULT_TIMEOUT) -> Check:
    """
    Is an Ollama server answering?

    Preferred probe is the HTTP API (`GET /api/version`), which is what the
    Python client the agent uses actually talks to. If that is unavailable we
    fall back to the CLIs (`ollama list`, then `ollama ps`) so a working daemon
    behind a non-default socket is still recognised.
    """
    host = normalize_host(host)

    try:
        payload = _http_json(f"{host}/api/version", timeout)
        version = str(payload.get("version") or "unknown")
        return Check("Ollama daemon", True,
                     f"version {version} via {host}/api/version", method="http")
    except urllib.error.HTTPError as exc:
        http_problem = f"HTTP {exc.code} from {host}/api/version"
    except urllib.error.URLError as exc:
        http_problem = f"cannot reach {host} ({exc.reason})"
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        http_problem = f"{type(exc).__name__}: {exc}"

    for args, label in (("list", "ollama list"), ("ps", "ollama ps")):
        result = _run_cli(args, timeout=max(timeout, 10.0))
        if result is not None and result[0] == 0:
            return Check("Ollama daemon", True,
                         f"answering (verified with '{label}'; HTTP probe said: "
                         f"{http_problem})", method="cli")

    cli = _run_cli("list", timeout=max(timeout, 10.0))
    if cli is None:
        hint = ("the 'ollama' binary was not found either. Install Ollama: "
                "curl -fsSL https://ollama.com/install.sh | sh "
                "(macOS: brew install ollama, Windows: winget install Ollama.Ollama)")
    else:
        hint = (f"start it with 'ollama serve' (or 'systemctl --user start ollama'), "
                f"then re-run. CLI said: {(cli[2] or cli[1]).strip().splitlines()[:1]}")

    return Check("Ollama daemon", False, f"no server at {host} ({http_problem})",
                 hint=hint, method="none")


def check_ollama_model(model: str, host: str,
                       timeout: float = DEFAULT_TIMEOUT,
                       daemon_ok: bool = True) -> Check:
    """Is `model` actually pulled? Checks /api/tags first, then `ollama list`."""
    host = normalize_host(host)
    name = f"Ollama model '{model}'"

    if daemon_ok:
        try:
            payload = _http_json(f"{host}/api/tags", timeout)
            installed = [
                str(item.get("name") or item.get("model") or "")
                for item in payload.get("models", [])
                if isinstance(item, dict)
            ]
            installed = [tag for tag in installed if tag]
            if model_matches(installed, model):
                return Check(name, True,
                             f"present ({len(installed)} model(s) installed)", method="http")
            listing = ", ".join(sorted(installed)) or "none"
            return Check(name, False, f"not pulled; installed: {listing}",
                         hint=f"pull it with:  ollama pull {model}", method="http")
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError,
                OSError, json.JSONDecodeError) as exc:
            http_problem = f"{type(exc).__name__}: {exc}"
    else:
        http_problem = "daemon not reachable"

    cli = _run_cli("list", timeout=max(timeout, 10.0))
    if cli is not None and cli[0] == 0:
        installed = _cli_model_names(cli[1])
        if model_matches(installed, model):
            return Check(name, True, "present (via 'ollama list')", method="cli")
        listing = ", ".join(sorted(installed)) or "none"
        return Check(name, False, f"not pulled; installed: {listing}",
                     hint=f"pull it with:  ollama pull {model}", method="cli")

    return Check(name, False, f"could not list models ({http_problem})",
                 hint=f"run 'ollama list' yourself, then:  ollama pull {model}",
                 method="none")


# ---------------------------------------------------------------------------
#  orchestration
# ---------------------------------------------------------------------------

def run_checks(model: str, host: str | None = None,
               timeout: float = DEFAULT_TIMEOUT) -> list[Check]:
    """Run the daemon check, then the model check. Never raises."""
    host = normalize_host(host)
    try:
        daemon = check_ollama_daemon(host, timeout)
    except Exception as exc:  # noqa: BLE001 - a health check must never crash
        daemon = Check("Ollama daemon", False,
                       f"unexpected error: {type(exc).__name__}: {exc}",
                       hint="report this, then start ollama manually: ollama serve")
    try:
        model_check = check_ollama_model(model, host, timeout, daemon_ok=daemon.ok)
    except Exception as exc:  # noqa: BLE001
        model_check = Check(f"Ollama model '{model}'", False,
                            f"unexpected error: {type(exc).__name__}: {exc}",
                            hint=f"manual fix:  ollama pull {model}")
    return [daemon, model_check]


def report(checks: Sequence[Check], stream=None,
           color: bool | None = None) -> bool:
    """Print readable PASS/FAIL lines. Returns True when everything passed."""
    stream = stream or sys.stdout
    if color is None:
        color = stream.isatty() and not os.environ.get("NO_COLOR")

    ok = True
    stream.write("Ollama health check\n")
    stream.write("-" * 68 + "\n")
    for check in checks:
        ok = ok and check.ok
        tag = _color("1;32", f"[{PASS}]", color) if check.ok else _color("1;31", f"[{FAIL}]", color)
        stream.write(f"{tag} {check.name}: {check.detail}\n")
        if not check.ok and check.hint:
            stream.write(f"       -> {check.hint}\n")
    stream.write("-" * 68 + "\n")
    if ok:
        stream.write(_color("1;32", "All checks passed - the agent can run.\n", color))
    else:
        stream.write(_color("1;31", "Health check FAILED - fix the items above.\n", color))
    stream.flush()
    return ok


def run(model: str = DEFAULT_MODEL, host: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        emit: Callable[[Check], None] | None = None,
        quiet: bool = False) -> bool:
    """
    Library entry point used by news_agent.py.

    Returns True when both the daemon and the model check pass. When `emit` is
    given, each check is handed to that callback instead of stdout, so the agent
    can route it through its own logger.
    """
    if emit is not None or quiet:
        checks = run_checks(model, host, timeout)
        if emit is not None:
            for check in checks:
                emit(check)
        elif not quiet:
            report(checks)
        return all(check.ok for check in checks)

    return report(run_checks(model, host, timeout))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="health_check.py",
        description="Verify the Ollama daemon and model before the news agent runs.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model tag to verify (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"Ollama base URL (default: {DEFAULT_HOST})")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--quiet", action="store_true",
                        help="print nothing, only set the exit code")
    args = parser.parse_args(argv)

    try:
        checks = run_checks(args.model, args.host, args.timeout)
        ok = all(check.ok for check in checks)
        if not args.quiet:
            report(checks)
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 - never show a traceback from here
        if not args.quiet:
            print(f"[{FAIL}] health check crashed unexpectedly: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            print("       -> try manually:  ollama serve   then   ollama list",
                  file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
