#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AI NEWS AGENT  --  100% LOCAL & FREE  (no paid APIs, no cloud keys)
================================================================================

 What it does
 ------------
 1. Finds the news      : DuckDuckGo (free, no API key) + `site:` operators
 2. Fetches the article : trafilatura (local) -> Jina AI Reader (free) fallback
 3. Thinks about it     : local LLM through Ollama (summarize / 140 words / EN+ES)
 4. Speaks it           : edge-tts (free Microsoft neural voices, no key) -> MP3
 5. Transcribes media   : faster-whisper (local, offline) for audio/video input
 6. Saves everything    : JSON posts + MP3 voiceovers in ./output

 Output shape
 ------------
   4 daily posts (one per category: USA, China, Venezuela, AI). Each post
   bundles 5 trending stories, every story with a full bilingual EN+ES script
   (<=140 words per language), plus ONE combined voiceover MP3 per post.

 Files
 -----
   news_agent.py      - this agent
   health_check.py    - mandatory pre-flight (Ollama daemon + model) - keep it
                        next to news_agent.py
   requirements.txt   - Python dependencies
   run.sh             - convenience launcher using ./.venv

 Stack (all free, all local)
 ---------------------------
   trafilatura        - boilerplate-free article text extraction (local)
   faster-whisper     - local speech-to-text (CTranslate2 implementation)
   edge-tts           - free neural TTS, no API key, needs internet only for TTS
   duckduckgo-search  - free web search (newer package name: ddgs)
   ollama             - local LLM runtime (llama3 / mistral / qwen ...)
   requests           - HTTP for the Jina reader fallback
   feedparser         - RSS fallback when web search returns nothing
   ffmpeg (system)    - required by faster-whisper to decode audio/video


================================================================================
 requirements.txt
 ================================================================================
 # --- save the block below as requirements.txt ---
 trafilatura>=1.8.0
 faster-whisper>=1.0.0
 edge-tts>=6.1.9
 duckduckgo-search>=6.1.0
 ddgs>=9.0.0
 ollama>=0.3.0
 requests>=2.31.0
 feedparser>=6.0.11
 # --- end of requirements.txt ---

 Manual install:
     python3 -m venv .venv && source .venv/bin/activate
     pip install -r requirements.txt


================================================================================
 HOW TO INSTALL OLLAMA AND PULL A MODEL
 ================================================================================
 Linux (one-liner):
     curl -fsSL https://ollama.com/install.sh | sh
 macOS:
     brew install ollama   # or download the .dmg from https://ollama.com
 Windows:
     winget install Ollama.Ollama     # or the installer from https://ollama.com

 Start the server (it is started automatically by this script if needed):
     ollama serve

 Pull a model (>= 3B params recommended for bilingual output):
     ollama pull llama3          # ~4.7 GB, best quality/size trade-off
     ollama pull mistral         # ~4.1 GB, faster, slightly weaker
     ollama pull qwen2.5:3b      # ~1.9 GB, great for weak machines
     ollama pull llama3.2:3b     # ~2.0 GB, good compromise

 Verify:
     ollama list
     ollama run llama3 "Hola, escribe una frase corta en espanol."

 NOTE: quality of the bilingual output is directly tied to the model size.
       7B-8B models follow the strict 140-word format far better than 3B ones.


================================================================================
 MANDATORY PRE-FLIGHT HEALTH CHECK
================================================================================
 health_check.py must sit next to this file. Before doing any work, the agent
 verifies the Ollama daemon (GET /api/version, falling back to `ollama list` /
 `ollama ps`) and that the requested model is actually pulled. Every check
 prints a readable PASS/FAIL line; a failure stops the run with an actionable
 message instead of a traceback.

 Run it on its own at any time:
     python3 health_check.py
     python3 health_check.py --model qwen2.5:3b


================================================================================
 USAGE
 ================================================================================
   OUTPUT SHAPE: 4 daily posts (one per category), each bundling 5 trending
   stories with a full EN+ES script each, plus ONE combined voiceover per post.

   python3 news_agent.py                       # 4 daily posts x 5 stories
   python3 news_agent.py --skip-setup          # do not touch pip/system
   python3 news_agent.py --model mistral       # use another local model
   python3 news_agent.py --categories USA AI   # only some categories
   python3 news_agent.py --limit 3             # 3 trending stories per post
   python3 news_agent.py --no-audio            # skip edge-tts
   python3 news_agent.py --es-audio            # also voice the Spanish scripts
   python3 news_agent.py --media interview.mp3   # transcribe + format a file
================================================================================
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# Standard library only at import time. Every third-party module is imported
# lazily AFTER the self-healing setup phase, so the setup code can run on a
# bare Python installation with nothing installed.
# ----------------------------------------------------------------------------
import argparse
import asyncio
import importlib
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, NoReturn
from urllib.parse import urlparse

# ============================================================================
#  CONSTANTS
# ============================================================================

APP_NAME = "news-agent"
DEFAULT_MODEL = os.environ.get("NEWS_AGENT_MODEL", "llama3")
DEFAULT_OUTPUT_DIR = Path(os.environ.get("NEWS_AGENT_OUTPUT", "output"))
WHISPER_MODEL_SIZE = os.environ.get("NEWS_AGENT_WHISPER", "base")

# Exactly 0.3 -> factual, obeys strict word counts instead of being creative.
LLM_TEMPERATURE = 0.3
LLM_NUM_CTX = 8192
WORDS_PER_LANGUAGE = 140

# --- Free search backends: DuckDuckGo with `site:` operators ---------------
CATEGORIES: dict[str, dict[str, Any]] = {
    "USA": {
        # `query` is the spec's site: restriction, kept verbatim. `topic` is
        # appended at search time so a Reuters Americas-desk story cannot land
        # in the USA feed (the operators alone carry no topical signal).
        "query": "site:reuters.com OR site:apnews.com",
        "topic": "United States",
        "keywords": ["united states", "u.s.", "america", "washington", "white house",
                     "congress", "senate", "trump", "federal"],
        "voice": "en-US-GuyNeural",
        "rss": [
            "https://feeds.bbci.co.uk/news/world/us_canada/rss.xml",
            "https://feeds.npr.org/1001/rss.xml",
        ],
    },
    "China": {
        "query": "site:reuters.com OR site:scmp.com",
        "topic": "China",
        "keywords": ["china", "chinese", "beijing", "shanghai", "hong kong",
                     "xi jinping", "sino-", "ccp"],
        "voice": "en-US-AriaNeural",
        "rss": [
            "https://feeds.bbci.co.uk/news/world/asia/china/rss.xml",
            "https://www.scmp.com/rss/4/feed",
        ],
    },
    "Venezuela": {
        "query": "site:efectococuyo.com OR site:runrun.es",
        "topic": "Venezuela",
        "keywords": ["venezuela", "venezolan", "caracas", "maduro", "chavismo",
                     "nicolás", "oposición"],
        # English voice on purpose: the English script must be read in English.
        "voice": "en-US-JennyNeural",
        "rss": [
            "https://efectococuyo.com/feed/",
            "https://runrun.es/feed/",
        ],
    },
    "AI": {
        "query": "site:techcrunch.com OR site:theverge.com",
        "topic": "artificial intelligence",
        "keywords": ["artificial intelligence", " ai", "ai ", "llm", "openai",
                     "anthropic", "chatbot", "machine learning", "neural", "gpt"],
        "voice": "en-US-ChristopherNeural",
        "rss": [
            "https://techcrunch.com/category/artificial-intelligence/feed/",
            "https://www.theverge.com/rss/index.xml",
        ],
    },
}

SPANISH_VOICE = "es-MX-JorgeNeural"
DEFAULT_EN_VOICE = "en-US-GuyNeural"   # safety net: English text -> English voice

# --- Free RSS fallback (used only when DuckDuckGo yields nothing) ----------
GENERIC_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
]

# --- Hybrid fetch tuning ---------------------------------------------------
MIN_LOCAL_CHARS = 500          # below this, trafilatura result is "not good enough"
JINA_READER = "https://r.jina.ai/"
HTTP_TIMEOUT = 35

# ============================================================================
#  THE AGENT'S SYSTEM PROMPT (verbatim, as specified)
# ============================================================================
SYSTEM_PROMPT = """You are an expert AI News Aggregator and Translator.

YOUR TASK:
Analyze the retrieved news articles (or transcribed audio) and format them for social media. You must output results in BOTH English and Spanish.

OUTPUT FORMATTING (STRICT 140-WORD LIMIT PER LANGUAGE):
For EVERY single news story, generate a multi-platform update in English, then translate to Spanish. The TOTAL word count for EACH language version MUST NOT EXCEED 140 words.

Format each story exactly like this:

[CATEGORY] - [Headline]

🇺🇸 ENGLISH VERSION:
🎬 Video Script (TikTok/Reels):
- Hook: [1 punchy sentence]
- The News: [2-3 sentences of core facts]
- Why It Matters: [2 sentences on impact]
📝 Feed Caption: [2-3 sentences, 2 emojis, 3 hashtags]

🇪🇸 VERSIÓN EN ESPAÑOL:
🎬 Guion de Video (TikTok/Reels):
- Gancho: [1 frase impactante]
- La Noticia: [2-3 oraciones con los hechos]
- Por Qué Importa: [2 oraciones sobre el impacto]
📝 Caption para Feed: [2-3 oraciones, 2 emojis, 3 hashtags]

RULES:
- Only use facts retrieved from the provided text. Do not hallucinate.
- Keep the tone engaging and conversational.
- Strictly enforce the 140-word maximum limit per language.
"""

MARKER_EN = "🇺🇸"
MARKER_ES = "🇪🇸"
MARKER_EN_ALT = "ENGLISH VERSION"
MARKER_ES_ALT = "VERSIÓN EN ESPAÑOL"

# Minimum words for a language block to be considered a real script (see
# block_quality): blocks below this are retried, never narrated.
MIN_BLOCK_WORDS = 25

# Chat-template tokens that small local models occasionally leak.
CONTROL_TOKEN_RE = re.compile(
    r"<\|?\s*(?:im_start|im_end|endoftext|eot_id|tool_call|tool_response|"
    r"start_header_id|end_header_id)\s*\|?>|</?\s*tool_call\s*>",
    re.IGNORECASE,
)

# A block that is only a placeholder, e.g. "[NONE]" or "NADA".
PLACEHOLDER_RE = re.compile(r"^[\W_]*(none|nada|n/?a|sin datos|nothing)[\W_]*$", re.IGNORECASE)

# Regex used for the "real" word count (ignores pure emoji/punctuation tokens,
# still counts hashtags and words because they contain alphanumerics).
WORD_RE = re.compile(r"[^\s]*[0-9A-Za-zÀ-ÖØ-öø-ÿ][^\s]*", re.UNICODE)
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U0000FE0F\U00002B00-\U00002BFF]+",
    re.UNICODE,
)

# ============================================================================
#  LOGGING  (clear, actionable, never silent)
# ============================================================================
_COLOR = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


class _Formatter(logging.Formatter):
    STYLES = {
        logging.DEBUG: ("90", "DBG"),
        logging.INFO: ("36", " .. "),
        logging.WARNING: ("33", "WARN"),
        logging.ERROR: ("31", "FAIL"),
        logging.CRITICAL: ("41;97", "CRIT"),
    }

    def format(self, record: logging.LogRecord) -> str:
        color, tag = self.STYLES.get(record.levelno, ("0", "INFO"))
        stamp = datetime.now().strftime("%H:%M:%S")
        head = _c(color, f"[{tag}]")
        msg = record.getMessage()
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text and record.levelno >= logging.ERROR:
            msg = f"{msg}\n{textwrap.indent(record.exc_text, '        ')}"
        return f"{_c('90', stamp)} {head} {msg}"


log = logging.getLogger(APP_NAME)


def setup_logging(verbose: bool = False) -> None:
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter())
    log.handlers.clear()
    log.addHandler(handler)
    log.propagate = False


def die(message: str, code: int = 1) -> NoReturn:
    """Log an actionable error and exit gracefully - never crash silently."""
    log.error(message)
    log.error("Exiting. Re-run with --skip-setup to manage dependencies by hand.")
    sys.exit(code)


# ============================================================================
#  SECTION 1 -- SELF-HEALING SETUP  (the "find-skill" discovery layer)
# ============================================================================
#
#  This is the local, dependency-free equivalent of the `find-skills` workflow:
#  discover the capability -> is it present? -> if not, acquire it -> verify it.
#  Nothing here needs third-party code, so it can repair the very environment
#  it is running in.
#
#  Every entry is described by: import name, pip name, and a human hint.
# ----------------------------------------------------------------------------

PYTHON_SKILLS: list[dict[str, Any]] = [
    # (module to import, pip requirement, why we need it, extra aliases)
    {"module": "trafilatura", "pip": "trafilatura",
     "purpose": "local article text extraction"},
    {"module": "faster_whisper", "pip": "faster-whisper",
     "purpose": "local speech-to-text (Whisper)"},
    {"module": "edge_tts", "pip": "edge-tts",
     "purpose": "free neural text-to-speech (MP3)"},
    {"module": "ddgs", "pip": "ddgs", "aliases": ["duckduckgo_search"],
     "purpose": "free web search (new package name)"},
    {"module": "duckduckgo_search", "pip": "duckduckgo-search",
     "aliases": ["ddgs"], "purpose": "free web search (legacy package name)"},
    {"module": "ollama", "pip": "ollama", "purpose": "local LLM runtime client"},
    {"module": "requests", "pip": "requests", "purpose": "HTTP (Jina fallback)"},
    {"module": "feedparser", "pip": "feedparser", "purpose": "RSS fallback"},
]


def _module_available(module: str) -> bool:
    """True if `module` can be imported right now (spec discovery, no import)."""
    try:
        if importlib.util.find_spec(module) is not None:
            return True
    except (ImportError, ValueError):
        pass
    # Already-imported modules (or vendored ones) also count.
    return module in sys.modules


def _in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _pip_install(requirement: str, quiet: bool = True) -> bool:
    """
    Install one pip requirement, escalating strategy until it works.

    Escalation only makes sense for *environment* problems (PEP 668
    "externally-managed-environment", permission errors). Build failures and
    network errors are reported immediately instead of being retried three
    times, which would just re-run a slow native build that cannot succeed.
    """
    base = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    if quiet:
        base.append("-q")
    strategies = [
        base + ["--upgrade", requirement],                 # normal path
        base + ["--user", "--upgrade", requirement],       # user site-packages
        base + ["--break-system-packages", requirement],   # Debian/Fedora opt-in
    ]
    if _in_virtualenv():
        strategies = strategies[:1]                        # venv: no escalation needed

    permission_markers = (
        "externally-managed-environment",
        "permission denied",
        "not writeable",
        "write access to",
        "consider using a virtual environment",
        "is not owned by the current user",
    )

    for cmd in strategies:
        log.debug("running: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            log.warning("pip timed out while installing %s", requirement)
            continue
        except OSError as exc:
            log.warning("could not run pip: %s", exc)
            continue

        if proc.returncode == 0:
            return True

        stderr = (proc.stderr or "") + (proc.stdout or "")
        needs_more_rights = any(m in stderr.lower() for m in permission_markers)
        if not needs_more_rights:
            # Build/network failure: retrying with another flag will not help.
            log.warning("pip failed to install %s (not a permissions issue)", requirement)
            log.debug("pip said: %s", stderr.strip()[-500:])
            if "failed to build" in stderr.lower() or "subprocess-exited-with-error" in stderr:
                log.error("  %s has no prebuilt wheel for Python %s and its source "
                          "build failed.", requirement, platform.python_version())
                log.error("  Common fix: use Python 3.11-3.12, or install the native "
                          "dependency (e.g. ffmpeg dev headers / rust toolchain) first.")
            return False
        log.debug("pip needs more rights for %s: %s", requirement, stderr.strip()[-300:])

    return False


def find_skill(skill: dict[str, Any], auto_install: bool = True) -> bool:
    """
    Discover one capability and heal the environment if needed.

    Returns True when the capability is usable. Never raises, never exits -
    the caller decides how fatal a missing skill is.
    """
    names = [skill["module"]] + list(skill.get("aliases", []))
    if any(_module_available(n) for n in names):
        log.debug("skill ready: %s (%s)", skill["module"], skill["purpose"])
        return True

    log.warning("missing skill: %s  ->  %s", skill["module"], skill["purpose"])
    if not auto_install:
        log.error("  install it with:  %s -m pip install %s", sys.executable, skill["pip"])
        return False

    log.info("installing %s ...", skill["pip"])
    if not _pip_install(skill["pip"]):
        log.error("could not auto-install %s", skill["pip"])
        log.error("  manual fix:  %s -m pip install %s", sys.executable, skill["pip"])
        if platform.system() == "Linux" and not _in_virtualenv():
            log.error("  (on Fedora/Debian use a venv: python3 -m venv .venv && "
                      "source .venv/bin/activate)")
        return False

    # Re-check: pip may have installed the alias package instead.
    importlib.invalidate_caches()
    for n in names:
        if _module_available(n):
            log.info("skill installed and verified: %s", skill["module"])
            return True

    log.error("%s installed but still not importable - check for build errors "
              "with:  %s -m pip install --force-reinstall %s",
              skill["pip"], sys.executable, skill["pip"])
    return False


def ensure_ffmpeg(auto_install: bool = True) -> bool:
    """ffmpeg is a hard requirement of faster-whisper's audio decoding."""
    exe = shutil.which("ffmpeg")
    if exe:
        log.debug("skill ready: ffmpeg (%s)", exe)
        return True

    log.warning("missing skill: ffmpeg  ->  required to decode audio/video for Whisper")
    managers = [
        (["apt-get", "install", "-y", "ffmpeg"], "apt-get", "Debian/Ubuntu"),
        (["dnf", "install", "-y", "ffmpeg"], "dnf", "Fedora/RHEL"),
        (["pacman", "-S", "--noconfirm", "ffmpeg"], "pacman", "Arch"),
        (["zypper", "install", "-y", "ffmpeg"], "zypper", "openSUSE"),
        (["apk", "add", "ffmpeg"], "apk", "Alpine"),
        (["brew", "install", "ffmpeg"], "brew", "macOS/Homebrew"),
    ]

    if auto_install:
        for args, mgr, distro in managers:
            if not shutil.which(mgr):
                continue
            cmd = list(args)
            if os.geteuid() != 0 and mgr != "brew" and shutil.which("sudo"):
                cmd = ["sudo"] + cmd
            log.info("installing ffmpeg with %s (%s) ...", mgr, distro)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            except (subprocess.TimeoutExpired, OSError) as exc:
                log.warning("ffmpeg install via %s failed: %s", mgr, exc)
                continue
            if proc.returncode == 0 and shutil.which("ffmpeg"):
                log.info("skill installed and verified: ffmpeg")
                return True
            log.debug("ffmpeg install via %s returned %s", mgr, proc.returncode)

    log.error("ffmpeg is required but unavailable.")
    log.error("  install it manually with one of:")
    log.error("    Debian/Ubuntu : sudo apt-get install -y ffmpeg")
    log.error("    Fedora        : sudo dnf install -y ffmpeg")
    log.error("    Arch          : sudo pacman -S ffmpeg")
    log.error("    macOS         : brew install ffmpeg")
    log.error("    Windows       : winget install Gyan.FFmpeg")
    log.error("  then re-run:  python3 %s --skip-setup", Path(sys.argv[0]).name)
    return False


# ---------------------------------------------------------------------------
#  Ollama: binary + server + model discovery
# ---------------------------------------------------------------------------

def _ollama_binary() -> str | None:
    return shutil.which("ollama")


def _ollama_models(client: Any) -> list[str]:
    """Return installed model tags. Tolerates old dict and new object APIs."""
    try:
        listing = client.list()
    except Exception:  # noqa: BLE001 - library raises many different types
        return []
    entries = listing.get("models", []) if isinstance(listing, dict) else getattr(listing, "models", []) or []
    tags: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            tag = entry.get("model") or entry.get("name") or ""
        else:
            tag = getattr(entry, "model", None) or getattr(entry, "name", None) or ""
        if tag:
            tags.append(str(tag))
    return tags


def _model_present(tags: Iterable[str], wanted: str) -> bool:
    wanted = wanted.strip()
    base = wanted.split(":")[0]
    for tag in tags:
        if tag == wanted or tag == f"{wanted}:latest":
            return True
        if ":" not in wanted and tag.split(":")[0] == base:
            return True
    return False


def ensure_ollama_ready(client: Any, model: str, auto_install: bool = True) -> bool:
    """
    Guarantee: (a) an ollama server answers, (b) `model` is pulled.
    Starts `ollama serve` in the background when the daemon is down.
    """
    if _ollama_server_up(client):
        log.debug("skill ready: ollama server")
    else:
        log.warning("missing skill: ollama server (not answering)")
        if not _ollama_binary():
            log.error("the ollama binary was not found either.")
            log.error("  install Ollama first:  curl -fsSL https://ollama.com/install.sh | sh")
            log.error("  macOS: brew install ollama    Windows: winget install Ollama.Ollama")
            return False
        if not auto_install:
            log.error("  start it manually:  ollama serve")
            return False

        log.info("starting 'ollama serve' in the background ...")
        try:
            subprocess.Popen(
                [_ollama_binary() or "ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            log.error("could not start ollama: %s", exc)
            return False

        for _ in range(30):
            time.sleep(1)
            if _ollama_server_up(client):
                log.info("skill ready: ollama server (started)")
                break
        else:
            log.error("ollama server did not come up within 30s. Start it manually: ollama serve")
            return False

    tags = _ollama_models(client)
    if _model_present(tags, model):
        log.debug("skill ready: ollama model %s", model)
        return True

    log.warning("missing skill: ollama model '%s' is not pulled", model)
    if not auto_install:
        log.error("  pull it manually:  ollama pull %s", model)
        return False

    log.info("pulling model '%s' (this can take several minutes) ...", model)
    try:
        last = 0.0
        for progress in client.pull(model, stream=True):
            status = (progress.get("status") if isinstance(progress, dict)
                      else getattr(progress, "status", "")) or ""
            now = time.time()
            if status and now - last > 5:
                log.info("  %s", status)
                last = now
    except Exception as exc:  # noqa: BLE001
        log.error("pull failed: %s", exc)
        log.error("  manual fix:  ollama pull %s", model)
        return False

    if _model_present(_ollama_models(client), model):
        log.info("skill installed and verified: ollama model %s", model)
        return True

    log.error("model '%s' still not listed after pull. Check free disk space "
              "and run:  ollama list", model)
    return False


def _ollama_server_up(client: Any) -> bool:
    try:
        client.list()
        return True
    except Exception:  # noqa: BLE001
        return False


def run_health_check(model: str) -> bool:
    """
    Mandatory pre-flight: verify the Ollama daemon and the requested model
    before the agent does any real work.

    Delegates to health_check.py, prints one readable PASS/FAIL line per check
    and never raises, so a broken environment produces a clean, actionable error
    instead of a stack trace.
    """
    log.info("=== HEALTH CHECK (mandatory) ===")
    try:
        import health_check
    except Exception as exc:  # noqa: BLE001
        log.error("could not import health_check.py (%s)", exc)
        log.error("  health_check.py must sit next to news_agent.py - restore it "
                  "and re-run.")
        return False

    try:
        checks = health_check.run_checks(model)
    except Exception as exc:  # noqa: BLE001 - the check itself must never crash us
        log.error("the health check failed unexpectedly: %s: %s",
                  type(exc).__name__, exc)
        log.error("  try manually:  ollama serve   then   ollama list")
        return False

    for check in checks:
        # The logger already tags the level ([...] FAIL/WARN), so do not repeat it.
        if check.ok:
            log.info("health check PASS: %s - %s", check.name, check.detail)
        else:
            log.error("health check FAIL: %s - %s", check.name, check.detail)
            if check.hint:
                log.error("                    -> %s", check.hint)
    return all(check.ok for check in checks)


# ---------------------------------------------------------------------------
#  The setup phase as a whole
# ---------------------------------------------------------------------------

def run_setup(model: str, auto_install: bool = True) -> dict[str, bool]:
    """
    Walk every required skill, heal what is missing, report what is still broken.
    Returns a {skill: ok} map so the caller can decide what is fatal.
    """
    log.info("=== SETUP PHASE  (auto-install: %s) ===", "on" if auto_install else "off")
    log.info("python %s on %s", platform.python_version(), platform.system())
    if not auto_install:
        log.info("--skip-setup given: dependencies will only be inspected, not changed")

    if sys.version_info < (3, 9):
        die(f"Python 3.9+ is required, found {platform.python_version()}")

    results: dict[str, bool] = {}
    for skill in PYTHON_SKILLS:
        results[skill["module"]] = find_skill(skill, auto_install)

    # Search works if either backend landed; report the pair as one capability.
    search_ok = results.get("ddgs") or results.get("duckduckgo_search")
    if not search_ok:
        log.error("no DuckDuckGo search backend available - news gathering will "
                  "fall back to RSS only")

    results["ffmpeg"] = ensure_ffmpeg(auto_install)

    # faster-whisper is optional: only --media needs it.
    if not results.get("faster_whisper"):
        log.warning("transcription is unavailable: --media will not work. "
                    "Everything else still runs.")
    if not results.get("edge_tts"):
        log.warning("edge-tts is unavailable: run with --no-audio, or fix the "
                    "install shown above.")

    core_ok = all([
        results.get("trafilatura"),
        results.get("ollama"),
        results.get("requests"),
        results.get("edge_tts"),
    ])
    if not core_ok:
        die("core skills are missing (trafilatura / ollama / requests / edge-tts). "
            "Fix the errors above and re-run.")

    # Import now that everything is present.
    missing = load_runtime_modules()
    if missing:
        die("required modules still missing after setup: "
            f"{', '.join(missing)} -> {sys.executable} -m pip install -r requirements.txt")
    if ollama is None:
        die("the ollama Python client could not be imported - see the errors above.")
    if not ensure_ollama_ready(ollama, model, auto_install):
        die(f"Ollama model '{model}' is not available. See the errors above.")

    log.info("=== SETUP COMPLETE ===")
    return results


# ============================================================================
#  SECTION 2 -- RUNTIME MODULE LOADING (lazy, after setup)
# ============================================================================
trafilatura: Any = None
edge_tts: Any = None
ollama: Any = None
requests: Any = None
feedparser: Any = None
WhisperModel: Any = None
DDGS: Any = None
SEARCH_BACKEND = "none"


# Modules the agent genuinely cannot run without (edge-tts is only needed when
# audio generation is requested; faster-whisper only for --media).
CORE_MODULES: tuple[str, ...] = ("trafilatura", "requests", "ollama")

# module name -> pip name, so error messages always give the right fix.
PIP_HINT: dict[str, str] = {s["module"]: s["pip"] for s in PYTHON_SKILLS}
PIP_HINT["faster_whisper"] = "faster-whisper"


def _safe_import(name: str, required: bool = False) -> Any:
    """
    Import a module without ever raising. Missing REQUIRED modules produce an
    actionable error (and are reported back to the caller); optional ones only
    warn, so the agent can degrade gracefully instead of crashing.
    """
    try:
        module = importlib.import_module(name)
        log.debug("imported %s", name)
        return module
    except Exception as exc:  # noqa: BLE001 - ImportError, OSError (bad .so), SyntaxError...
        hint = PIP_HINT.get(name, name)
        if required:
            log.error("cannot import required module '%s': %s", name, exc)
            log.error("  fix it with:  %s -m pip install %s", sys.executable, hint)
        else:
            log.warning("optional module '%s' unavailable (%s) - feature disabled", name, exc)
            log.debug("  to enable it:  %s -m pip install %s", sys.executable, hint)
        return None


def load_runtime_modules(force: bool = False) -> list[str]:
    """
    Import every third-party module the agent needs. Idempotent and safe.

    Returns the list of REQUIRED modules that are still missing, so main() can
    exit gracefully with a clear message instead of a traceback.
    """
    global trafilatura, edge_tts, ollama, requests, feedparser
    global WhisperModel, DDGS, SEARCH_BACKEND

    def _need(module: Any, name: str) -> Any:
        if module is not None and not force:
            return module
        return _safe_import(name, required=name in CORE_MODULES)

    trafilatura = _need(trafilatura, "trafilatura")
    requests = _need(requests, "requests")
    ollama = _need(ollama, "ollama")
    feedparser = _need(feedparser, "feedparser")      # optional (RSS fallback)
    edge_tts = _need(edge_tts, "edge_tts")            # optional (--no-audio)

    if WhisperModel is None and (force or _module_available("faster_whisper")):
        module = _safe_import("faster_whisper")
        WhisperModel = getattr(module, "WhisperModel", None) if module else None

    if DDGS is None:
        for candidate in ("ddgs", "duckduckgo_search"):
            if not (force or _module_available(candidate)):
                continue
            module = _safe_import(candidate)
            backend = getattr(module, "DDGS", None) if module else None
            if backend is not None:
                DDGS, SEARCH_BACKEND = backend, candidate
                log.debug("search backend: %s", candidate)
                break

    return [m for m in CORE_MODULES if globals().get(m) is None]


# ============================================================================
#  SECTION 3 -- SEARCH  (DuckDuckGo, free, no API key)
# ============================================================================

def _ddg_search(query: str, max_results: int, timelimit: str, mode: str) -> list[dict[str, str]]:
    """
    One DuckDuckGo call, normalised to [{title, url, snippet}].
    mode="text" -> ddgs.text(), mode="news" -> ddgs.news() (news index, dates).
    """
    if DDGS is None:
        return []

    attempts: list[dict[str, Any]] = []
    if mode == "news":
        attempts = [{"max_results": max_results, "timelimit": timelimit},
                    {"max_results": max_results},
                    {}]
    else:
        attempts = [{"max_results": max_results, "timelimit": timelimit},
                    {"max_results": max_results},
                    {}]

    for kwargs in attempts:
        try:
            with DDGS() as ddgs:
                call = ddgs.news if mode == "news" else ddgs.text
                raw = list(call(query, **kwargs))
        except Exception as exc:  # noqa: BLE001 - ddgs raises many types, also rate-limits
            log.debug("[search:%s] attempt failed for %s: %s", mode, query, exc)
            time.sleep(1.5)
            continue

        results: list[dict[str, str]] = []
        for item in raw:
            url = (item.get("href") or item.get("url") or "").strip()
            if not url.startswith("http"):
                continue
            results.append({
                "title": (item.get("title") or "").strip(),
                "url": url,
                "snippet": (item.get("body") or item.get("snippet") or "").strip(),
            })
        if results:
            return dedupe_by_url(results)[:max_results]
    return []


def search_news(query: str, max_results: int = 6, timelimit: str = "w") -> list[dict[str, str]]:
    """
    DuckDuckGo text search restricted with `site:` operators (free, no API key).
    Homepage/section URLs are filtered out. Returns [] on any failure.
    """
    results = _ddg_search(f"{query} latest news", max_results, timelimit, mode="text")
    return filter_articles(results)[:max_results]


def search_news_feed(query: str, max_results: int = 6, timelimit: str = "w") -> list[dict[str, str]]:
    """
    DuckDuckGo *news* endpoint - a different index that returns dated articles.
    Used as the second retrieval layer when plain text search only gives
    homepages. The `site:` syntax is often rejected here, so a plain keyword
    query is tried too.
    """
    plain = re.sub(r"site:\s*", "", query)   # "reuters.com OR apnews.com"
    plain = " ".join(plain.split()) or query

    for candidate in (f"{query} latest news", f"{plain} news"):
        results = filter_articles(_ddg_search(candidate, max_results, timelimit, mode="news"))
        if results:
            return results[:max_results]
    return []


def gather_sources(config: dict[str, Any], limit: int) -> tuple[list[dict[str, str]], str]:
    """
    Try every free retrieval layer and MERGE what they return, because
    DuckDuckGo regularly gives only one or two usable links for a narrow
    `site:` query. Returns (candidates, "layer+layer" label).

    Layering matters: DuckDuckGo rate-limits and news sites rotate their RSS.
    """
    # Keep the spec's site: operators exactly, but add the topical term so the
    # results are actually about the category. Over-fetch candidates because
    # some will be dropped (off-topic, unfetchable, incomplete, already used).
    query = f"{config['query']} {config.get('topic', '')}".strip()
    keywords = config.get("keywords", [])
    wanted = max(limit * 3, 8)

    def news_layer() -> list[dict[str, str]]:
        results = search_news_feed(query, max_results=max(wanted, 10))
        if not results:
            return []
        relevant = [r for r in results if is_relevant(r, keywords)]
        if len(relevant) != len(results):
            log.debug("[gather] ddg-news: dropped %d off-topic result(s)",
                      len(results) - len(relevant))
        return relevant

    collected: list[dict[str, str]] = []
    used_layers: list[str] = []

    for layer, fetch in (
        ("ddg-text", lambda: search_news(query, max_results=max(wanted * 3, 10))),
        ("ddg-news", news_layer),
        ("rss", lambda: fetch_rss_fallback(config["rss"], max_results=wanted)),
    ):
        if len(collected) >= wanted:
            break
        try:
            results = fetch()
        except Exception as exc:  # noqa: BLE001
            log.warning("[gather:%s] failed: %s", layer, exc)
            continue
        if not results:
            log.info("[gather] layer '%s' yielded nothing", layer)
            continue
        used_layers.append(layer)
        collected = dedupe_by_url(collected + results)
        log.debug("[gather] %s -> %d unique candidate(s) so far",
                  layer, len(collected))

    # Generic world feeds are a true last resort: they are not category-scoped.
    if not collected:
        try:
            collected = dedupe_by_url(fetch_rss_fallback(GENERIC_FEEDS, max_results=wanted))
        except Exception as exc:  # noqa: BLE001
            log.warning("[gather:rss-generic] failed: %s", exc)
        if collected:
            used_layers.append("rss-generic")
            log.warning("[gather] falling back to generic world feeds")

    if not collected:
        return [], "none"

    # Strongest article-shaped candidates first, weak section pages last.
    collected.sort(key=lambda item: article_score(item["url"]), reverse=True)
    return collected[:max(wanted, limit)], "+".join(used_layers)


def fetch_rss_fallback(urls: Iterable[str], max_results: int = 4) -> list[dict[str, str]]:
    """Free RSS fallback so the agent still works when search is rate-limited."""
    if feedparser is None:
        return []
    out: list[dict[str, str]] = []
    for feed_url in urls:
        if len(out) >= max_results:
            break
        try:
            feed = feedparser.parse(feed_url)
        except Exception as exc:  # noqa: BLE001
            log.debug("[rss] %s failed: %s", feed_url, exc)
            continue
        entries = getattr(feed, "entries", [])
        if not entries:
            log.debug("[rss] %s returned no entries", feed_url)
            continue
        for entry in entries:
            link = (getattr(entry, "link", "") or "").strip()
            if not link.startswith("http"):
                continue
            out.append({
                "title": (getattr(entry, "title", "") or "").strip(),
                "url": link,
                "snippet": re.sub(r"<[^>]+>", " ",
                                  getattr(entry, "summary", "") or "").strip()[:400],
            })
            if len(out) >= max_results:
                break
    if out:
        log.debug("[rss] produced %d item(s)", len(out))
    return filter_articles(dedupe_by_url(out))[:max_results]


# URL shapes that are never news articles: homepages, sections, tag/author
# pages, feeds... Keeping them out saves fetch + LLM time on junk.
# Words that mean "this is a section page" when they are the LAST path segment
# (e.g. /news, /latest, /tag/politics). They must NOT be matched anywhere in the
# path: almost every article lives under /news/..., so a global match would
# reject the entire BBC/SCMP/Al Jazeera feeds.
SECTION_SEGMENTS = {
    "about", "contact", "search", "tag", "tags", "category", "categories",
    "author", "authors", "feed", "rss", "sitemap", "privacy", "terms",
    "newsletter", "subscribe", "login", "signin", "signup", "page",
    "latest", "news", "topics", "topic", "video", "videos", "podcast",
    "internacionales", "nacionales", "opinion", "deportes", "sucesos",
}

# Structural path segments that always mean a listing page, even when they are
# not the last segment (/tag/foo, /category/bar, /news/videos/baz). "news" is
# deliberately absent: /news/<article-slug> is where most articles live.
STRUCTURAL_SEGMENTS = {
    "tag", "tags", "category", "categories", "author", "authors", "search",
    "topics", "topic", "feed", "rss", "sitemap", "video", "videos", "podcast",
}

# Hosts that only serve newsletters / roundup links, never a single article.
NON_ARTICLE_HOSTS = ("newslink.", "newsletter.")


def looks_like_article(url: str) -> bool:
    """
    Cheap heuristic to reject homepages and section pages before we spend a
    fetch + LLM call on them. Anything uncertain is kept (false negatives are
    cheaper than losing every result).
    """
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return True

    if parsed.netloc.lower().startswith(NON_ARTICLE_HOSTS):
        return False                      # e.g. newslink.reuters.com

    segments = [s for s in parsed.path.split("/") if s]
    if not segments:                      # e.g. https://techcrunch.com/
        return False

    if any(s.lower() in STRUCTURAL_SEGMENTS for s in segments[:-1]):
        return False                      # /tag/foo, /category/bar, /videos/baz

    slug = segments[-1].lower()
    if slug in SECTION_SEGMENTS:          # e.g. /news, /latest, /tag/politics
        return False
    if len(parsed.path) < 8:
        return False
    # "/la-humanidad/" style single-segment paths are sections, not articles.
    if len(segments) == 1 and parsed.path.endswith("/"):
        return False
    # Real articles almost always carry a hyphenated slug or a numeric id.
    if "-" not in slug and not re.search(r"\d", slug):
        return False
    return True


def article_score(url: str) -> int:
    """
    Confidence that a URL is a real article, used to try the weak candidates
    last instead of rejecting them outright. Section pages such as
    reuters.com/world/us/new-york/ score low and only get used if a category
    cannot be filled any other way.
    """
    try:
        path = urlparse(url).path
    except Exception:  # noqa: BLE001
        return 1
    segments = [s for s in path.split("/") if s]
    if not segments:
        return 0
    slug = segments[-1]
    if re.search(r"\d", slug):          # /2026/10/16/slug or /1009086/slug
        return 3
    if slug.count("-") >= 3:            # long hyphenated headline slug
        return 3
    if slug.count("-") >= 1:
        return 2
    return 1


def filter_articles(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Keep only article-shaped URLs (homepages and section pages are dropped) and
    order them most-article-like first, so weak candidates are tried last.
    """
    good = [i for i in items if looks_like_article(i["url"])]
    if good and len(good) != len(items):
        log.debug("[search] dropped %d non-article URLs", len(items) - len(good))
    # Stable sort: search order is preserved between equally strong candidates.
    good.sort(key=lambda i: article_score(i["url"]), reverse=True)
    return good


def url_key(url: str) -> str:
    """Normalise a URL for duplicate detection."""
    return url.split("#")[0].rstrip("/").lower()


def headline_key(title: str) -> str:
    """
    Normalise a headline for duplicate detection.

    Different URLs can carry the same story (syndication, AMP, newsletter
    roundups), so a post is deduplicated by headline as well as by URL.
    """
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def dedupe_by_url(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in items:
        key = url_key(item["url"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def is_relevant(item: dict[str, str], keywords: list[str]) -> bool:
    """
    Cheap topical check against the title + snippet.

    Needed because the DuckDuckGo *news* endpoint silently ignores `site:`
    operators and the topical term: asking for "site:reuters.com OR
    site:scmp.com China" happily returns a Reuters Americas-desk story. The
    site: text search honours the operators, so this guard is only applied to
    the news layer.
    """
    if not keywords:
        return True
    haystack = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
    return any(keyword.lower() in haystack for keyword in keywords)


# ============================================================================
#  SECTION 4 -- HYBRID FETCH  (local first, Jina only as a fallback)
# ============================================================================

def fetch_article(url: str) -> str:
    """
    Get the readable body text of an article.

      1. LOCAL   : trafilatura.fetch_url + trafilatura.extract (comments off)
      2. FALLBACK: Jina AI Reader, free, no API key -> https://r.jina.ai/<url>

    Returns clean text, or "" - this function never raises.
    """
    text = _fetch_local(url)
    if len(text) >= MIN_LOCAL_CHARS:
        log.debug("[fetch] local OK (%d chars) %s", len(text), url)
        return text

    log.debug("[fetch] local gave %d chars, trying Jina reader: %s", len(text), url)
    remote = _fetch_jina(url)
    if len(remote) > len(text):
        log.debug("[fetch] Jina OK (%d chars) %s", len(remote), url)
        return remote

    log.warning("[fetch] no usable text for %s", url)
    return text  # may be a short-but-real local extraction, or ""


def _fetch_local(url: str) -> str:
    """Step 1 of the hybrid fetch: pure-local extraction with trafilatura."""
    if trafilatura is None:
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        extracted = trafilatura.extract(
            downloaded,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            deduplicate=True,
        )
        return normalize_text(extracted or "")
    except Exception as exc:  # noqa: BLE001
        log.debug("[fetch] trafilatura failed for %s: %s", url, exc)
        return ""


def _fetch_jina(url: str) -> str:
    """Step 2 of the hybrid fetch: free Jina AI Reader (no key required)."""
    if requests is None:
        return ""
    headers = {
        "User-Agent": f"{APP_NAME}/1.0 (+local; free-tier)",
        "Accept": "text/plain",
        "X-Return-Format": "text",
        "X-Timeout": "20",
    }
    try:
        response = requests.get(f"{JINA_READER}{url}", headers=headers, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return normalize_text(response.text)
    except Exception as exc:  # noqa: BLE001
        log.debug("[fetch] Jina failed for %s: %s", url, exc)
        return ""


def normalize_text(text: str) -> str:
    """Collapse whitespace, drop nav junk, cap length for the LLM context."""
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    text = text.replace("\r", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)      # markdown images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)   # markdown links
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)   # no trailing spaces per line
    text = re.sub(r"\n[ \t]+", "\n", text)   # no leading indentation per line
    return text.strip()


# ============================================================================
#  SECTION 5 -- LOCAL LLM  (Ollama, temperature 0.3)
# ============================================================================

def llm_chat(messages: list[dict[str, str]], model: str, temperature: float = LLM_TEMPERATURE) -> str:
    """One chat completion against the local Ollama server."""
    if ollama is None:
        return ""
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_ctx": LLM_NUM_CTX},
        )
    except Exception as exc:  # noqa: BLE001
        log.error("[llm] Ollama call failed: %s", exc)
        return ""
    return _response_text(response)


def _response_text(response: Any) -> str:
    """Support both dict-style (old) and object-style (new) ollama responses."""
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return (message.get("content") or "").strip()
        return (getattr(message, "content", "") or "").strip()
    message = getattr(response, "message", None)
    return ((getattr(message, "content", "") or "").strip() if message else "").strip()


def count_words(text: str) -> int:
    """
    Social-media style word count: counts tokens containing letters/digits
    (so emojis don't inflate the number, but hashtags do).
    """
    if not text:
        return 0
    return len(WORD_RE.findall(text))


def split_bilingual(output: str) -> tuple[str, str]:
    """
    Split the model output into (english_block, spanish_block).

    A "[CATEGORY] - Headline" banner is removed (the model often glues the
    🇺🇸 marker to the end of that line), and the English block is then given the
    standard "🇺🇸 ENGLISH VERSION:" label so both languages are self-labelled.
    """
    idx_es = output.find(MARKER_ES)
    if idx_es == -1:
        idx_es = output.find(MARKER_ES_ALT)
    if idx_es == -1:
        return _labelled(_strip_banners(output), MARKER_EN, MARKER_EN_ALT), ""

    english = _strip_banners(output[:idx_es])
    spanish = output[idx_es:].strip()
    return _labelled(english, MARKER_EN, MARKER_EN_ALT), spanish


def _labelled(block: str, marker: str, label: str) -> str:
    """Ensure a language block starts with its own '🇺🇸 ENGLISH VERSION:' label."""
    block = block.strip()
    if block and not block.lstrip().startswith(marker):
        return f"{marker} {label}:\n\n{block}"
    return block


def _is_banner(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("[") and "]" in stripped and len(stripped) < 200


def _strip_banners(text: str) -> str:
    """
    Drop leading/trailing '[CATEGORY] - Headline' banner lines. The
    '🇺🇸 ENGLISH VERSION:' label never starts with '[' so it is never touched.
    """
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and _is_banner(lines[0]):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    while lines and _is_banner(lines[-1]):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines).strip()


# --- strict-format compliance checks ---------------------------------------
EMOJI_COUNT_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]")
CAPTION_MARKERS = {"English": "Feed Caption", "Spanish": "Caption para Feed"}
REQUIRED_BULLETS = {
    "English": ["Hook", "The News", "Why It Matters"],
    "Spanish": ["Gancho", "La Noticia", "Por Qué Importa"],
}


def validate_format(block: str, language: str) -> list[str]:
    """
    Check one language block against the mandated output format. Strict 140
    words, 2 emojis and 3 hashtags in the caption, and all four required
    bullets. Small models (3B) miss these often - we surface it instead of
    pretending the output is compliant.
    """
    warnings: list[str] = []
    if not block.strip():
        return [f"{language}: empty output"]

    words = count_words(block)
    if words > WORDS_PER_LANGUAGE:
        warnings.append(f"{language}: {words} words > {WORDS_PER_LANGUAGE} limit")

    for bullet in REQUIRED_BULLETS[language]:
        if not re.search(rf"^[\s\-•*]*{re.escape(bullet)}\s*:", block,
                         re.MULTILINE | re.IGNORECASE):
            warnings.append(f"{language}: missing bullet '{bullet}'")

    marker = CAPTION_MARKERS[language]
    index = block.rfind(marker)
    if index == -1:
        warnings.append(f"{language}: missing '{marker}'")
    else:
        caption = block[index + len(marker):]
        emojis = len(EMOJI_COUNT_RE.findall(caption))
        hashtags = len(re.findall(r"#\w+", caption))
        if emojis < 2:
            warnings.append(f"{language}: caption has {emojis} emoji(s), format asks for 2")
        if hashtags < 3:
            warnings.append(f"{language}: caption has {hashtags} hashtag(s), format asks for 3")
    return warnings


# Model tags that are small enough to struggle with the strict output format.
SMALL_MODEL_RE = re.compile(r"[:\-](0\.5|1|1\.5|2|3|4|6)b\b", re.IGNORECASE)


def report_format_warnings(record: dict[str, Any], model: str = "") -> list[str]:
    """Validate both languages of a story record and log what is non-compliant."""
    warnings = validate_format(record.get("english", ""), "English")
    # Always validate Spanish too: a missing Spanish block is itself a failure.
    warnings += validate_format(record.get("spanish", ""), "Spanish")
    for warning in warnings:
        log.warning("[format] %s", warning)
    if warnings:
        log.warning("[format] %d issue(s) for '%s'.", len(warnings),
                    record.get("headline", "")[:50])
        # Only suggest a bigger model when the current one is actually small.
        if SMALL_MODEL_RE.search(model or ""):
            log.warning("[format] Tip: '%s' is a small model and often drops "
                        "hashtags/emojis; a 7B+ model (llama3, mistral) follows "
                        "the strict format far more reliably.", model)
    return warnings


def enforce_word_limit(text: str, language: str, model: str) -> str:
    """
    Hard 140-word guarantee. If the model overshoots, ask it (same temp 0.3)
    to compress its own text without losing facts, then accept the best try.
    """
    if count_words(text) <= WORDS_PER_LANGUAGE:
        return text

    log.warning("[llm] %s block is %d words (> %d) - compressing",
                language, count_words(text), WORDS_PER_LANGUAGE)
    prompt = (
        f"The text below is a {language} social-media news script. It has "
        f"{count_words(text)} words. Rewrite it in {WORDS_PER_LANGUAGE} words or FEWER, "
        "keeping every fact, the same labels/bullets, the same emojis and the same "
        "hashtags. Do not add new information. Output only the rewritten text.\n\n"
        f"{text}"
    )
    compressed = llm_chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        model=model,
    )
    compressed = compressed.strip()
    if compressed and count_words(compressed) < count_words(text):
        if count_words(compressed) <= WORDS_PER_LANGUAGE:
            log.info("[llm] %s block compressed to %d words", language, count_words(compressed))
        return compressed
    log.error("[llm] %s block still exceeds %d words", language, WORDS_PER_LANGUAGE)
    return text


def clean_llm_output(text: str) -> str:
    """Strip chat-template tokens small models sometimes leak into the answer."""
    if not text:
        return ""
    text = CONTROL_TOKEN_RE.sub(" ", text)
    text = re.sub(r"^\s*(assistant|user|system)\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    return normalize_text(text)


def block_quality(english: str, spanish: str) -> tuple[bool, str]:
    """
    Reject degenerate answers before they reach the JSON or the microphone.

    Small models sometimes answer with "[NONE]"/"[NADA]" or drop one language
    entirely. Those blocks are worse than useless: they get narrated into an MP3
    and published as if they were real. Treat them as a failure and retry.
    """
    checks = (("English", english), ("Spanish", spanish))
    for language, block in checks:
        if not block.strip():
            return False, f"{language} section missing"
        if PLACEHOLDER_RE.match(block):
            return False, f"{language} section is a placeholder"
        words = count_words(block)
        if words < MIN_BLOCK_WORDS:
            return False, f"{language} section too short ({words} words)"
    return True, ""


def generate_script(category: str, title: str, source_text: str, model: str) -> dict[str, Any]:
    """
    The core agent step: feed one article to the local LLM with the system
    prompt and get back the bilingual 140-word multi-platform update.

    Two attempts maximum: if the model drops a language or answers with a
    placeholder, we ask again with an explicit correction, then keep whichever
    attempt was most complete.
    """
    body = source_text.strip()
    if len(body) > 12000:  # keep the prompt inside num_ctx
        body = body[:12000] + "\n[...truncated...]"

    user_prompt = textwrap.dedent(f"""
        CATEGORY: {category}
        SOURCE HEADLINE: {title or "(no headline available)"}

        RETRIEVED TEXT (the ONLY source of facts you may use):
        \"\"\"
        {body}
        \"\"\"

        Produce the bilingual output now, exactly in the required format,
        using {MARKER_EN} and {MARKER_ES} section markers.
        Maximum {WORDS_PER_LANGUAGE} words PER LANGUAGE. Spanish must be a real
        translation of the English version, not a summary of it.
    """).strip()

    best: tuple[int, str, str, bool, str] | None = None
    for attempt in (1, 2):
        raw = clean_llm_output(llm_chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_prompt}],
            model=model,
        ))

        if not raw:
            log.error("[llm] empty answer for: %s", title[:70])
            continue

        english, spanish = split_bilingual(raw)
        complete, reason = block_quality(english, spanish)
        score = min(count_words(english), count_words(spanish))

        if best is None or score > best[0]:
            best = (score, english, spanish, complete, reason)
        if complete:
            break

        log.warning("[llm] attempt %d incomplete (%s) for: %s", attempt, reason, title[:60])
        user_prompt += textwrap.dedent(f"""

            CORRECTION REQUIRED: your previous answer was rejected because the
            {reason}. Reply again with the COMPLETE bilingual output. Fill in
            EVERY bullet in BOTH languages with real facts from the text above.
            Never write placeholders such as [NONE] or [NADA].
        """)

    if best is None:
        return {"english": "", "spanish": "", "word_counts": {"en": 0, "es": 0},
                "complete": False, "quality_issue": "no LLM output"}

    _, english, spanish, complete, reason = best
    english = enforce_word_limit(english, "English", model)
    if spanish:
        spanish = enforce_word_limit(spanish, "Spanish", model)

    if not complete:
        log.error("[llm] giving up on a complete answer for '%s' (%s)", title[:60], reason)

    return {
        "english": english,
        "spanish": spanish,
        "word_counts": {"en": count_words(english), "es": count_words(spanish)},
        "complete": complete,
        "quality_issue": "" if complete else reason,
    }


# ============================================================================
#  SECTION 6 -- TEXT-TO-SPEECH  (edge-tts: free, no API key)
# ============================================================================

# edge-tts has a per-request size limit; longer narration is chunked and stitched.
TTS_CHUNK_CHARS = 3000


def clean_for_tts(text: str) -> str:
    """Strip markers/emojis/hashtags so the voice reads like a news anchor."""
    if not text:
        return ""
    text = EMOJI_RE.sub(" ", text)
    text = re.sub(r"^[#>\-\*•]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    text = re.sub(r"#\w+", "", text)                       # hashtags
    # Drop section labels so the voice reads like a news anchor, not a script.
    text = re.sub(
        r"(?i)(video script[^:\n]*:|guion de video[^:\n]*:|feed caption\s*:|"
        r"caption para feed\s*:|english version\s*:|versión en español\s*:?)",
        " ", text,
    )
    text = re.sub(r"(?i)^(hook|the news|why it matters|gancho|la noticia|por qué importa)\s*:",
                  "", text, flags=re.MULTILINE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_for_tts(text: str, limit: int = TTS_CHUNK_CHARS) -> list[str]:
    """
    Break long narration into request-sized chunks at sentence boundaries.
    A daily post narrates 5 scripts, which is far more than edge-tts accepts
    in a single websocket request.
    """
    if len(text) <= limit:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        while len(sentence) > limit:                 # one monstrous sentence
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(sentence[:limit].strip())
            sentence = sentence[limit:]
        if current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def concat_mp3(parts: list[Path], dest: Path) -> bool:
    """
    Stitch MP3 parts into one file.

    Prefers an ffmpeg stream copy (no re-encode); falls back to raw byte
    concatenation, which is still playable because every part comes from the
    same encoder with identical sample rate and bitrate.
    """
    if not parts:
        return False
    if len(parts) == 1:
        shutil.move(str(parts[0]), str(dest))
        return dest.exists() and dest.stat().st_size > 0

    if shutil.which("ffmpeg"):
        listing = dest.with_suffix(".concat.txt")
        listing.write_text(
            "".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "concat", "-safe", "0", "-i", str(listing),
                 "-c", "copy", str(dest)],
                capture_output=True, text=True, timeout=600, check=False,
            )
            if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                return True
            log.warning("[tts] ffmpeg concat failed (%s) - using byte concat",
                        (proc.stderr or "").strip()[:200])
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("[tts] ffmpeg concat error (%s) - using byte concat", exc)
        finally:
            listing.unlink(missing_ok=True)

    with open(dest, "wb") as handle:
        for part in parts:
            handle.write(part.read_bytes())
    return dest.exists() and dest.stat().st_size > 0


def synthesize_speech(text: str, voice: str, dest: Path) -> bool:
    """
    Generate an MP3 voiceover with edge-tts (free Microsoft neural voices).

    Long text is chunked and stitched, so one call can narrate a whole daily
    post (5 scripts). Returns True on success. Never raises.
    """
    if edge_tts is None:
        log.warning("[tts] edge-tts unavailable, skipping %s", dest.name)
        return False

    spoken = clean_for_tts(text)
    if not spoken:
        log.warning("[tts] nothing to speak for %s", dest.name)
        return False

    chunks = split_for_tts(spoken)
    dest.parent.mkdir(parents=True, exist_ok=True)

    async def _save(chunk_text: str, path: Path) -> None:
        await edge_tts.Communicate(chunk_text, voice).save(str(path))

    if len(chunks) == 1:
        try:
            asyncio.run(_save(chunks[0], dest))
        except Exception as exc:  # noqa: BLE001
            log.error("[tts] edge-tts failed for %s: %s", dest.name, exc)
            log.error("     edge-tts needs internet for the Microsoft speech endpoint; "
                      "check your connection or re-run with --no-audio")
            return False
    else:
        log.info("[tts] %s: %d chunk(s) to narrate, then stitch",
                 dest.name, len(chunks))
        with tempfile.TemporaryDirectory(prefix="news-agent-tts-") as tmp:
            parts: list[Path] = []
            for number, chunk in enumerate(chunks, start=1):
                part = Path(tmp) / f"part{number:02d}.mp3"
                try:
                    asyncio.run(_save(chunk, part))
                except Exception as exc:  # noqa: BLE001
                    log.error("[tts] chunk %d/%d failed for %s: %s",
                              number, len(chunks), dest.name, exc)
                    return False
                if not part.exists() or part.stat().st_size == 0:
                    log.error("[tts] chunk %d/%d came back empty for %s",
                              number, len(chunks), dest.name)
                    return False
                parts.append(part)
            if not concat_mp3(parts, dest):
                log.error("[tts] could not stitch the chunks for %s", dest.name)
                return False

    if dest.exists() and dest.stat().st_size > 0:
        log.info("[tts] %s  (%s, %d KB)", dest.name, voice, dest.stat().st_size // 1024)
        return True
    log.error("[tts] produced an empty file: %s", dest)
    return False


# ============================================================================
#  SECTION 7 -- LOCAL TRANSCRIPTION  (faster-whisper, fully offline)
# ============================================================================
_WHISPER_CACHE: dict[str, Any] = {}


def get_whisper(model_size: str = WHISPER_MODEL_SIZE, device: str = "cpu") -> Any:
    """Load (and memoize) a Whisper model. Returns None when unavailable."""
    if WhisperModel is None:
        log.error("[whisper] faster-whisper is not installed. "
                  "Fix with: %s -m pip install faster-whisper", sys.executable)
        return None
    if shutil.which("ffmpeg") is None:
        log.error("[whisper] ffmpeg is required to decode media. Install it first.")
        return None

    key = f"{model_size}:{device}"
    if key in _WHISPER_CACHE:
        return _WHISPER_CACHE[key]

    # float16 only makes sense on a GPU; "auto" resolves to CPU most of the time.
    compute_type = "float16" if device == "cuda" else "int8"
    log.info("[whisper] loading model '%s' (%s/%s) - first run downloads weights",
             model_size, device, compute_type)
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001
        log.warning("[whisper] could not load on %s (%s), retrying on CPU", device, exc)
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as exc2:  # noqa: BLE001
            log.error("[whisper] model load failed: %s", exc2)
            return None

    _WHISPER_CACHE[key] = model
    return model


def transcribe_local_media(
    file_path: str | Path,
    model_size: str = WHISPER_MODEL_SIZE,
    device: str = "cpu",
    language: str | None = None,
) -> str:
    """
    Transcribe a local audio/video file with faster-whisper (offline).
    Used for interviews, press briefings, voice notes, screen recordings.
    Returns the transcript, or "" on failure.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        log.error("[whisper] file not found: %s", path)
        return ""

    model = get_whisper(model_size, device)
    if model is None:
        return ""

    log.info("[whisper] transcribing %s ...", path.name)
    try:
        segments, info = model.transcribe(
            str(path),
            language=language,
            beam_size=5,
            vad_filter=True,                                    # skip silence
            vad_parameters={"min_silence_duration_ms": 600},
        )
        chunks = []
        for segment in segments:  # generator: transcription happens here
            text = (segment.text or "").strip()
            if text:
                chunks.append(text)
                log.debug("[whisper] %5.1fs-%5.1fs  %s",
                          segment.start, segment.end, text[:80])
    except Exception as exc:  # noqa: BLE001
        log.error("[whisper] transcription failed: %s", exc)
        return ""

    transcript = normalize_text(" ".join(chunks))
    log.info("[whisper] transcribed %s: %d chars (detected language: %s)",
             path.name, len(transcript), getattr(info, "language", "?"))
    return transcript


# ============================================================================
#  SECTION 8 -- THE EXECUTION FLOW
# ============================================================================

def process_story(
    category: str,
    result: dict[str, str],
    model: str,
) -> dict[str, Any] | None:
    """
    Fetch -> LLM for a single story. Returns the JSON record or None.

    Audio is deliberately NOT generated here: one voiceover is produced for the
    whole daily post (see build_post) instead of one file per story.
    """
    log.info("[%s] %s", category, (result["title"] or result["url"])[:80])

    article = fetch_article(result["url"])
    if len(article) < 200:
        article = normalize_text(f"{result.get('snippet', '')}\n{article}")
    if len(article) < 120:
        log.warning("[%s] not enough text, skipping %s", category, result["url"])
        return None

    script = generate_script(category, result["title"], article, model)
    if not script.get("english"):
        log.warning("[%s] the model produced no usable script, skipping", category)
        return None

    record: dict[str, Any] = {
        "category": category,
        "headline": result["title"],
        "url": result["url"],
        "fetched_chars": len(article),
        "word_counts": script["word_counts"],
        "english": script["english"],
        "spanish": script["spanish"],
        "complete": script.get("complete", True),
        "quality_issue": script.get("quality_issue", ""),
    }
    if not record["complete"]:
        log.error("[%s] script for '%s' is incomplete (%s) - it is saved for "
                  "inspection but will not be narrated", category, result["title"][:50],
                  record["quality_issue"])
    record["format_warnings"] = report_format_warnings(record, model)
    return record


def build_post(
    category: str,
    config: dict[str, Any],
    layer: str,
    effective_query: str,
    records: list[dict[str, Any]],
    model: str,
    output_dir: Path,
    make_audio: bool,
    es_audio: bool,
    date_str: str,
) -> dict[str, Any]:
    """
    Assemble a daily post: N trending stories plus ONE combined voiceover.

    The voiceover concatenates every story script in the post, so a 5-story
    post is a single MP3 per language rather than 5 separate files.
    """
    post_id = f"{category.lower()}_daily_{date_str}_news"
    post: dict[str, Any] = {
        "post_id": post_id,
        "category": category,
        "title": f"{category} Daily Roundup - {date_str}",
        "date": date_str,
        "query": config["query"],
        "effective_query": effective_query,
        "source_layer": layer,
        "story_count": len(records),
        "stories": records,
        "audio_en": None,
        "audio_es": None,
    }

    speakable = [r for r in records if count_words(r.get("english", "")) >= MIN_BLOCK_WORDS]
    if not make_audio or not speakable:
        return post

    # The English script is always voiced by an English neural voice.
    voice = config.get("voice", DEFAULT_EN_VOICE)
    en_voice = voice if str(voice).startswith("en-") else DEFAULT_EN_VOICE
    english = "\n\n".join(r["english"] for r in speakable)
    en_path = output_dir / f"{post_id}_en.mp3"
    if synthesize_speech(english, en_voice, en_path):
        post["audio_en"] = str(en_path)

    if es_audio:
        spanish = "\n\n".join(r["spanish"] for r in speakable
                              if count_words(r.get("spanish", "")) >= MIN_BLOCK_WORDS)
        if spanish:
            es_path = output_dir / f"{post_id}_es.mp3"
            if synthesize_speech(spanish, SPANISH_VOICE, es_path):
                post["audio_es"] = str(es_path)

    return post


def slugify(text: str, max_length: int = 48) -> str:
    text = EMOJI_RE.sub("", text or "")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:max_length] or "story"


def build_report(
    model: str,
    categories: list[str],
    limit: int,
    output_dir: Path,
    make_audio: bool,
    es_audio: bool,
    media_file: str | None,
    whisper_model: str,
    whisper_device: str,
) -> dict[str, Any]:
    date_str = datetime.now().strftime("%Y%m%d")
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": APP_NAME,
        "date": date_str,
        "model": model,
        "temperature": LLM_TEMPERATURE,
        "max_words_per_language": WORDS_PER_LANGUAGE,
        "stories_per_post": limit,
        "posts": [],
        "stack": {
            "search": SEARCH_BACKEND,
            "fetch": "trafilatura -> r.jina.ai",
            "llm": "ollama (local)",
            "tts": "edge-tts (free)",
            "stt": "faster-whisper (local)",
        },
    }

    # ---- optional: format a local audio/video file as its own post ---------
    if media_file:
        transcript = transcribe_local_media(media_file, whisper_model, whisper_device)
        if transcript:
            script = generate_script("Transcription", Path(media_file).name, transcript, model)
            record = {
                "category": "Transcription",
                "headline": f"Local media: {Path(media_file).name}",
                "url": str(Path(media_file).resolve()),
                "transcript_chars": len(transcript),
                "transcript": transcript,
                "word_counts": script.get("word_counts", {"en": 0, "es": 0}),
                "english": script.get("english", ""),
                "spanish": script.get("spanish", ""),
                "complete": script.get("complete", True),
                "quality_issue": script.get("quality_issue", ""),
            }
            record["format_warnings"] = report_format_warnings(record, model)
            source = str(Path(media_file).resolve())
            report["posts"].append({
                "post_id": f"transcription_daily_{date_str}",
                "category": "Transcription",
                "title": f"Transcription - {Path(media_file).name}",
                "date": date_str,
                "query": source,
                "effective_query": source,
                "source_layer": "local-file",
                "story_count": 1,
                "stories": [record],
                "audio_en": None,
                "audio_es": None,
            })

    # ---- the 4 daily posts (one per category) -----------------------------
    used_urls: set[str] = set()
    used_headlines: set[str] = set()
    for category in categories:
        config = CATEGORIES[category]
        effective_query = f"{config['query']} {config.get('topic', '')}".strip()
        log.info("--- CATEGORY: %s  (query: %s) ---", category, effective_query)

        def empty_post(reason: str, src: str = "") -> dict[str, Any]:
            return {
                "post_id": f"{category.lower()}_daily_{date_str}",
                "category": category,
                "title": f"{category} Daily Roundup - {date_str}",
                "date": date_str,
                "query": config["query"],
                "effective_query": effective_query,
                "source_layer": src,
                "story_count": 0,
                "stories": [],
                "audio_en": None,
                "audio_es": None,
                "error": reason,
            }

        results, layer = gather_sources(config, limit)
        if not results:
            log.error("[%s] no sources found on any layer, skipping category", category)
            report["posts"].append(empty_post("no sources found"))
            continue
        log.info("[%s] %d candidate source(s) via %s", category, len(results), layer)

        # Never tell the same story twice: not by URL, and not by headline
        # (syndicated copies and newsletter roundups reuse the same title).
        fresh = [
            r for r in results
            if url_key(r["url"]) not in used_urls
            and (not headline_key(r.get("title", ""))
                 or headline_key(r["title"]) not in used_headlines)
        ]
        dropped = len(results) - len(fresh)
        if dropped:
            log.debug("[%s] %d candidate(s) already used by an earlier post",
                      category, dropped)
        if not fresh:
            log.warning("[%s] every candidate was already used - skipping to "
                        "avoid duplicating a story", category)
            report["posts"].append(empty_post("all candidates already used", layer))
            continue

        stories: list[dict[str, Any]] = []
        skipped = 0
        for result in fresh:
            if len(stories) >= limit:
                break
            try:
                record = process_story(category, result, model)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad story must not kill the run
                log.error("[%s] story failed: %s", category, exc,
                          exc_info=log.isEnabledFor(logging.DEBUG))
                skipped += 1
                continue
            if not record:
                skipped += 1
                continue
            if not record["complete"]:
                # An incomplete bilingual script would degrade the post, so keep
                # looking for another story instead of publishing it.
                log.warning("[%s] dropping incomplete story '%s' (%s)",
                            category, record["headline"][:50], record["quality_issue"])
                skipped += 1
                continue
            hkey = headline_key(record["headline"])
            if hkey and hkey in used_headlines:
                # Same story, different URL (syndication, newsletter roundup).
                log.warning("[%s] dropping duplicate headline '%s'",
                            category, record["headline"][:50])
                skipped += 1
                continue
            used_urls.add(url_key(record["url"]))
            if hkey:
                used_headlines.add(hkey)
            stories.append(record)

        if len(stories) < limit:
            log.warning("[%s] collected %d/%d stories for today's post "
                        "(%d candidate(s) unusable)", category, len(stories),
                        limit, skipped)

        post = build_post(category, config, layer, effective_query, stories,
                          model, output_dir, make_audio, es_audio, date_str)
        post["skipped_candidates"] = skipped
        report["posts"].append(post)
        log.info("[%s] post ready: %d stories%s", category, post["story_count"],
                 f" -> {post['audio_en']}" if post.get("audio_en") else "")

    return report


def write_report(report: dict[str, Any], output_dir: Path, json_path: Path | None = None) -> Path:
    """Write the combined daily report plus one standalone JSON per post."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if json_path is None:
        stamp = report.get("date") or datetime.now().strftime("%Y%m%d")
        json_path = output_dir / f"daily_{stamp}_news.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("saved daily report -> %s", json_path)

    # One file per post: drop it straight into a scheduler / content calendar.
    for post in report.get("posts", []):
        post_path = output_dir / f"{post['post_id']}.json"
        post_path.write_text(json.dumps(post, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        log.debug("saved post -> %s", post_path)

    return json_path


def print_summary(report: dict[str, Any], json_path: Path) -> None:
    posts = report.get("posts", [])
    total = sum(len(p.get("stories", [])) for p in posts)
    posts_with_audio = sum(1 for p in posts if p.get("audio_en"))
    issues = sum(len(s.get("format_warnings") or [])
                 for p in posts for s in p.get("stories", []))

    print()
    print(_c("1;36", "=" * 78))
    print(_c("1;36", f" {APP_NAME} - daily posts for {report.get('date')}"))
    print(_c("1;36", "=" * 78))
    print(f" model        : {report['model']}  (temperature {report['temperature']})")
    print(f" posts        : {len(posts)}  (target "
          f"{report.get('stories_per_post')} trending stories each)")
    print(f" stories      : {total}")
    print(f" post audio   : {posts_with_audio}")
    print(f" format issues: {issues}")
    print(f" daily report : {json_path}")
    print("-" * 78)

    for post in posts:
        stories = post.get("stories", [])
        heading = f" [{post['category']}] {post.get('story_count', len(stories))} stories"
        print(_c("1;33", heading))
        if not stories:
            print(f"   - ({post.get('error', 'no stories')})")
            continue
        for number, story in enumerate(stories, start=1):
            words = story.get("word_counts", {})
            count = len(story.get("format_warnings") or [])
            flag = f"  ({count} format issue(s))" if count else ""
            print(f"   {number}. {story['headline'][:60]}{_c('33', flag)}")
            print(f"      en {words.get('en', 0)}w / es {words.get('es', 0)}w")
        if post.get("audio_en"):
            print(f"   -> {post['audio_en']}")
        if post.get("audio_es"):
            print(f"   -> {post['audio_es']}")
    print(_c("1;36", "=" * 78))


# ============================================================================
#  SECTION 9 -- CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news_agent.py",
        description="100% local & free bilingual (EN/ES) AI news agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Builds 4 daily posts (one per category), each bundling 5 trending
            stories with a bilingual script and one combined voiceover.

            examples:
              python3 news_agent.py
              python3 news_agent.py --skip-setup --model mistral --limit 3
              python3 news_agent.py --categories Venezuela AI --es-audio
              python3 news_agent.py --media briefing.mp4 --whisper-model small
        """),
    )
    parser.add_argument("--skip-setup", action="store_true",
                        help="do not install/verify dependencies of any kind")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"local Ollama model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                        choices=list(CATEGORIES) + ["all"], metavar="CAT",
                        help="USA China Venezuela AI (default: all)")
    parser.add_argument("--limit", type=int, default=5,
                        help="trending stories per daily post (default: 5)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"folder for MP3 + JSON (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--json", default=None,
                        help="explicit JSON output path (default: <output-dir>/news_<ts>.json)")
    parser.add_argument("--no-audio", action="store_true",
                        help="skip edge-tts voiceover generation")
    parser.add_argument("--es-audio", action="store_true",
                        help="also generate the Spanish voiceover")
    parser.add_argument("--media", default=None,
                        help="transcribe a local audio/video file and format it first")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL_SIZE,
                        help=f"faster-whisper size: tiny/base/small/medium/large-v3 "
                             f"(default: {WHISPER_MODEL_SIZE})")
    parser.add_argument("--whisper-device", default="cpu", choices=["cpu", "cuda", "auto"],
                        help="device for faster-whisper (default: cpu)")
    parser.add_argument("--transcribe-only", action="store_true",
                        help="with --media: only print the transcript, no LLM/TTS")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} 1.0.0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    categories = list(CATEGORIES) if "all" in args.categories else args.categories
    output_dir = Path(args.output_dir).expanduser()
    # Each run's artifacts live in their own day-stamped subfolder so daily
    # posts (JSON + MP3s) never overwrite the previous day's.
    run_date = datetime.now().strftime("%Y%m%d")
    output_dir = output_dir / run_date

    log.info("=== %s: 100%% local AI news agent ===", APP_NAME)

    # ---------- 1. SETUP PHASE (find-skill style, self-healing) ------------
    if not args.skip_setup:
        run_setup(args.model, auto_install=True)
    else:
        log.info("=== SETUP PHASE SKIPPED (--skip-setup) ===")
        missing = load_runtime_modules()
        if missing:
            log.error("required modules are missing: %s", ", ".join(missing))
            log.error("  with --skip-setup nothing is installed automatically.")
            log.error("  install them yourself, then re-run:")
            log.error("      %s -m pip install -r requirements.txt", sys.executable)
            return 1

    if ollama is None:
        die(f"the ollama Python client is unavailable. "
            f"Install it:  {sys.executable} -m pip install ollama")

    # Audio is optional: degrade gracefully if edge-tts is not importable.
    if not args.no_audio and edge_tts is None:
        log.warning("edge-tts is not available -> continuing with --no-audio")
        args.no_audio = True

    # Make sure the LLM is reachable even in --skip-setup mode.
    if not ensure_ollama_ready(ollama, args.model, auto_install=not args.skip_setup):
        die(f"cannot use Ollama model '{args.model}'. Pull it with: ollama pull {args.model}")

    # ------------------------------------------------------------------
    # MANDATORY pre-flight: the daemon and the model must both be healthy
    # before the agent is allowed to proceed.
    # ------------------------------------------------------------------
    if not run_health_check(args.model):
        die(f"health check failed - Ollama daemon and model '{args.model}' must "
            f"both be ready. Fix the items above (or run: ollama serve && "
            f"ollama pull {args.model}) and re-run.")

    # ---------- optional: transcribe a local file only ---------------------
    if args.transcribe_only:
        if not args.media:
            die("--transcribe-only requires --media <file>", 2)
        transcript = transcribe_local_media(args.media, args.whisper_model, args.whisper_device)
        if not transcript:
            return 1
        print(transcript)
        return 0

    # ---------- 2..6. NEWS -> LLM -> AUDIO -> OUTPUT ----------------------
    try:
        report = build_report(
            model=args.model,
            categories=categories,
            limit=max(1, args.limit),
            output_dir=output_dir,
            make_audio=not args.no_audio,
            es_audio=args.es_audio,
            media_file=args.media,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
        )
    except KeyboardInterrupt:
        log.warning("interrupted by the user")
        return 130

    json_path = write_report(report, output_dir,
                             Path(args.json).expanduser() if args.json else None)
    print_summary(report, json_path)

    total = sum(len(p.get("stories", [])) for p in report.get("posts", []))
    if total == 0:
        log.error("no stories were produced. Try a smaller --limit, another "
                  "--model, or check your internet connection for search/fetch.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
