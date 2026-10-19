# AI News Agent — 100% local, 100% free

A bilingual (English + Spanish) social-media news agent that runs entirely on your
own machine. No OpenAI, no ElevenLabs, no HeyGen, no API keys — ever.

## What it does

It produces **4 daily posts — one per category** (USA, China, Venezuela, AI), and
each post bundles the **5 trending news stories** of that category, every story with
a full bilingual (EN + ES) script, plus **one combined voiceover per post**.

1. **Finds the news** with DuckDuckGo (`site:` operators, no API key)
2. **Fetches articles** with `trafilatura` locally, falling back to the free Jina AI Reader
3. **Writes bilingual scripts** with a local LLM through Ollama (`temperature=0.3`, 140-word cap per language)
4. **Records voiceovers** with `edge-tts` (free Microsoft neural voices)
5. **Transcribes local media** with `faster-whisper` (offline)
6. **Saves** the daily report, one JSON per post and every MP3 to `./output`

## Free stack

| Capability | Tool | Key needed |
|---|---|---|
| Search | DuckDuckGo (`ddgs` / `duckduckgo-search`) | no |
| Article extraction | `trafilatura` (+ `r.jina.ai` fallback) | no |
| LLM | Ollama, local model | no |
| Text-to-speech | `edge-tts` | no |
| Speech-to-text | `faster-whisper` | no |
| RSS fallback | `feedparser` | no |

## Install

Recommended interpreter: **Python 3.11 – 3.13** (3.12 is verified working).
On Python 3.14+ PyAV has no prebuilt wheel, so `--media` transcription cannot
be installed.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ollama (the LLM runtime)
curl -fsSL https://ollama.com/install.sh | sh     # Linux
brew install ollama                               # macOS
winget install Ollama.Ollama                      # Windows

ollama pull llama3        # ~4.7 GB, best bilingual quality
ollama pull qwen2.5:3b    # ~1.9 GB, good for weak machines
```

### Quick start on this machine (already configured)

```bash
./run.sh --es-audio                 # everything is installed and ollama is running
./run.sh --limit 2 --categories AI  # quick two-story test
```

`run.sh` just activates `.venv` and runs the agent with whatever arguments you pass.

`ffmpeg` is required by Whisper. The agent installs it automatically via
`apt-get` / `dnf` / `pacman` / `zypper` / `apk` / `brew` when it is missing.

> **Python version note:** `faster-whisper` depends on PyAV (`av`) and
> CTranslate2, which publish prebuilt wheels only for recent-but-not-bleeding-edge
> Python. On Python 3.15 the source build of `av` fails, so transcription
> (`--media`) is unavailable — everything else still works, and the agent fails
> fast with an actionable message instead of retrying a doomed build.
> Use **Python 3.11–3.13** if you need `--media`.

## Usage

```bash
python3 news_agent.py                          # 4 daily posts x 5 trending stories
python3 news_agent.py --skip-setup             # never install anything
python3 news_agent.py --model mistral          # another local model
python3 news_agent.py --categories Venezuela AI
python3 news_agent.py --limit 3 --es-audio     # 3 stories/post, Spanish voiceover too
python3 news_agent.py --no-audio               # scripts only
python3 news_agent.py --media briefing.mp4     # transcribe a file, then format it
python3 news_agent.py --media note.mp3 --transcribe-only
```

| Flag | Meaning |
|---|---|
| `--skip-setup` | bypass all auto-installation (manual control) |
| `--model` | Ollama model tag (default `llama3`) |
| `--categories` | `USA China Venezuela AI` |
| `--limit` | trending stories bundled into each daily post (default 5) |
| `--output-dir` | where MP3 + JSON go (default `output/`) |
| `--no-audio` / `--es-audio` | disable audio / also voice the Spanish scripts |
| `--media` | transcribe a local audio/video file first |
| `--whisper-model` | `tiny` `base` `small` `medium` `large-v3` |
| `-v` | verbose debug logging |

## Mandatory health check

`health_check.py` must sit next to `news_agent.py`. Before doing any work the
agent runs it and **refuses to continue if it fails**:

* **Ollama daemon** — probed via `GET /api/version`; falls back to `ollama list`
  then `ollama ps`.
* **Ollama model** — verified via `/api/tags`; falls back to `ollama list`.

Each check prints a readable `PASS`/`FAIL` line with an actionable hint, and a
failure exits cleanly instead of dumping a traceback. You can also run it by
itself, for example from a cron job or a pre-flight script:

```bash
python3 health_check.py
python3 health_check.py --model qwen2.5:3b --host 127.0.0.1:11434
```

## Self-healing setup

At startup the agent runs a `find-skill`-style discovery pass: it checks every
Python package, the `ffmpeg` binary, the Ollama daemon and the model, and installs
whatever is missing (`--skip-setup` turns this off). Failures always produce an
actionable message and a graceful exit instead of a traceback.

## Output

One run produces **4 daily posts** (one per category), 5 stories each = 20 stories,
plus 8 voiceovers (EN and ES per post). Everything lands in a day-stamped
subfolder so yesterday's run is never overwritten.

```
output/
└── 20261019/
    ├── daily_20261019.json              # the whole run: all 4 posts, EN + ES scripts,
    │                                    #   word counts, format warnings, audio paths
    ├── usa_daily_20261019.json          # one standalone JSON per daily post,
    ├── usa_daily_20261019_en.mp3        #   ready to drop into a scheduler
    ├── usa_daily_20261019_es.mp3        #   (ES file only with --es-audio)
    ├── china_daily_20261019.json
    ├── china_daily_20261019_en.mp3
    ├── china_daily_20261019_es.mp3
    ├── venezuela_daily_20261019.{json,mp3}
    └── ai_daily_20261019.{json,mp3}
```

Each `*_daily_*.json` looks like this:

```json
{
  "post_id": "usa_daily_20261019",
  "category": "USA",
  "date": "20261019",
  "story_count": 5,
  "stories": [
    { "headline": "...", "url": "...", "word_counts": {"en": 103, "es": 117},
      "english": "🇺🇸 ENGLISH VERSION: ...", "spanish": "🇪🇸 VERSIÓN EN ESPAÑOL: ...",
      "complete": true, "format_warnings": [] }
  ],
  "audio_en": "output/usa_daily_20261019_en.mp3",
  "audio_es": "output/usa_daily_20261019_es.mp3"
}
```

The post voiceover narrates **all 5 stories** of the post in one MP3. Because that is
far longer than edge-tts accepts in a single request, the narration is split at
sentence boundaries and stitched back together (with `ffmpeg` when available).

Only **complete** bilingual stories enter a post: if a model drops a language, that
candidate is discarded and the agent keeps looking, so a post never ships half a
script. A story URL is also never reused across two categories.

## Troubleshooting

**`TypeError: open() got an unexpected keyword argument 'metadata_errors'`**
faster-whisper 1.2.x calls `av.open(..., metadata_errors="ignore")`, but PyAV 19
removed that keyword. Keep PyAV below 19 (already pinned in `requirements.txt`):

```bash
pip install "av<19"
```

**Ollama: "could not connect to ollama server"**
The agent starts `ollama serve` for you when needed. For a permanent service that
survives reboots, use a *user* unit (no root required):

```bash
mkdir -p ~/.config/systemd/user
# ~/.config/systemd/user/ollama.service with ExecStart=/usr/local/bin/ollama serve
systemctl --user enable --now ollama
loginctl enable-linger "$USER"    # start at boot without logging in
```

The official installer also creates a system-wide unit that runs as
`User=ollama` with `HOME=/usr/share/ollama`. If that directory does not exist,
the unit fails in a restart loop and never serves anything. Either create it:

```bash
sudo mkdir -p /usr/share/ollama/.ollama && sudo chown -R ollama:ollama /usr/share/ollama
```

or disable it and rely on the user unit above:

```bash
sudo systemctl disable --now ollama
```

Note that models pulled as your own user live in `~/.ollama/models`, which the
system-wide service cannot see unless you point `OLLAMA_MODELS` at it.

**`--media` says "faster-whisper is not installed"**
Only affects transcription. Check your Python version first — see the install
section above.

## Model quality note

The strict format (140 words per language, 2 emojis and 3 hashtags in each caption)
is a real constraint on small models. Measured on the same four categories:

| Model | Format issues per story | Notes |
|---|---|---|
| `qwen2.5:3b` | 3–5 | misses bullets, captions, sometimes a whole language |
| `llama3` (8B) | ~2 | usually just the third hashtag |

Every story is validated before it is saved or narrated, and non-compliance is
reported in `format_warnings` in the JSON, so nothing fails silently. If a model
drops a language entirely, the agent retries once with a correction prompt, and
refuses to narrate a block that is too short to be a real script.

## Category relevance

The `site:` operators alone carry no topical signal, so the agent appends a topic
term to each query (`site:reuters.com OR site:scmp.com China`). The DuckDuckGo
*news* endpoint ignores both `site:` and the topic term, so results from that
layer are additionally filtered against per-category keywords — this is what
stops a Reuters *Americas* story from appearing in the China feed. The same URL
is also never reused across two categories.
