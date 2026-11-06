"""Job-search agent: a hand-written tool-calling loop over any OpenAI-compatible LLM API.

Usage:
    python agent.py                 # interactive chat
    python agent.py "evaluate this job: <url or pasted text>"

Config (env vars):
    LLM_API_KEY   your provider's API key
    LLM_BASE_URL  e.g. https://generativelanguage.googleapis.com/v1beta/openai/  (Gemini)
                       https://api.groq.com/openai/v1                           (Groq)
                       https://openrouter.ai/api/v1                             (OpenRouter)
    LLM_MODEL     a model name from that provider

Put your resume in resume.md (see resume.sample.md).
"""
import json
import os
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import tracker

load_dotenv(Path(__file__).parent / ".env")

RESUME_PATH = Path(__file__).parent / "resume.md"
MAX_TURNS = 10
MAX_FETCH_CHARS = 12000

SYSTEM_PROMPT = """You are a personal job-search assistant for one person.

When given a job (pasted text or URL; if a URL, call fetch_url):
1. Call read_resume to get the candidate's resume.
2. Score fit 0-100 and justify it in 3-5 lines: strongest matches, real gaps.
3. Suggest 3 resume bullet rewrites that are TRUTHFUL. Reword and reorder what is
   already in the resume to match the posting; never invent experience or metrics.
4. Draft a short (under 100 words) outreach note to the hiring manager or recruiter.
5. If the fit score is 60+ or the user asks, call save_job to track it.

Be concise and direct. Flag red flags in the posting (vague role, unrealistic
requirements, no salary range where one is expected). When asked about the pipeline,
use list_jobs and update_job. Only use tools when needed."""


# ---------- tools ----------

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def read_resume() -> str:
    if not RESUME_PATH.exists():
        return "resume.md not found. Ask the user to create it."
    return RESUME_PATH.read_text()


def fetch_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Error: url must start with http:// or https://"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "jobhunt-agent/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:  # network errors are returned to the model, not raised
        return f"Error fetching page: {e}. Ask the user to paste the job text instead."
    parser = _TextExtractor()
    parser.feed(html)
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    if len(text) < 200:
        return "Page had almost no readable text (probably needs JavaScript). Ask the user to paste the job text."
    return text[:MAX_FETCH_CHARS]


def save_job(company: str, role: str, url: str = "", fit_score: int | None = None,
             notes: str = "") -> str:
    return json.dumps(tracker.add_job(company, role, url, fit_score, notes))


def list_jobs(status: str = "") -> str:
    return json.dumps(tracker.list_jobs(status or None), indent=2)


def update_job(job_id: str, status: str = "", notes: str = "") -> str:
    try:
        job = tracker.update_job(job_id, status or None, notes or None)
    except ValueError as e:
        return f"Error: {e}"
    return json.dumps(job) if job else "No job with that id."


TOOL_FUNCS = {
    "read_resume": read_resume,
    "fetch_url": fetch_url,
    "save_job": save_job,
    "list_jobs": list_jobs,
    "update_job": update_job,
}


def _tool(name, description, props=None, required=None):
    fn = {"name": name, "description": description}
    # Ensure properties is always a dictionary, even if empty, 
    # to avoid strict schema validation errors on some API gateways.
    params = {"type": "object", "properties": props or {}}
    if required:
        params["required"] = required
    fn["parameters"] = params
    return {"type": "function", "function": fn}


TOOL_SPECS = [
    _tool("read_resume", "Read the candidate's resume."),
    _tool("fetch_url", "Fetch a web page and return its readable text.",
          {"url": {"type": "string"}}, ["url"]),
    _tool("save_job", "Save a job to the tracker.",
          {"company": {"type": "string"}, "role": {"type": "string"},
           "url": {"type": "string"}, "fit_score": {"type": "integer"},
           "notes": {"type": "string"}}, ["company", "role"]),
    _tool("list_jobs", "List tracked jobs, best fit first. Optional status filter; omit for all.",
          {"status": {"type": "string", "enum": tracker.STATUSES}}),
    _tool("update_job", "Update a tracked job's status and/or add a note.",
          {"job_id": {"type": "string"},
           "status": {"type": "string", "enum": tracker.STATUSES},
           "notes": {"type": "string"}}, ["job_id"]),
]


def run_tool(name: str, raw_args: str) -> str:
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"Error: unknown tool {name}"
    try:
        args = json.loads(raw_args or "{}")
        return str(func(**args))
    except Exception as e:  # bad args from the model get reported back so it can retry
        return f"Error running {name}: {e}"


# ---------- the agent loop ----------

def run_turn(client, model: str, messages: list) -> None:
    """Run the model until it stops calling tools (or MAX_TURNS), printing its replies."""
    for _ in range(MAX_TURNS):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=TOOL_SPECS, tool_choice="auto",
            )
        except Exception as e:
            # If it's an OpenAI API error, print the response body for debugging
            if hasattr(e, "body") and e.body:
                print(f"\n[API Error Details]: {json.dumps(e.body, indent=2)}", file=sys.stderr)
            raise
            
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if msg.content:
            print(msg.content)
        if not msg.tool_calls:
            return
        for call in msg.tool_calls:
            print(f"  [tool: {call.function.name}]", file=sys.stderr)
            result = run_tool(call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    print("(stopped: too many tool steps)")
    

def make_client() -> tuple[OpenAI, str]:
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not api_key or not model:
        sys.exit("Set LLM_API_KEY and LLM_MODEL (and LLM_BASE_URL for non-OpenAI providers). See README.")
    return OpenAI(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL")), model


def main() -> None:
    client, model = make_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if len(sys.argv) > 1:
        messages.append({"role": "user", "content": " ".join(sys.argv[1:])})
        run_turn(client, model, messages)
        return
    print("Job-search agent. Paste a job URL/description, or ask about your pipeline. Ctrl+C to quit.")
    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt:
            messages.append({"role": "user", "content": prompt})
            run_turn(client, model, messages)


if __name__ == "__main__":
    main()