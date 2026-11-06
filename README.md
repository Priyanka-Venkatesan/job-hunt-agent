# Job-Search Agent

A personal agent that evaluates job postings against your resume, drafts tailored
bullets and an outreach note, and tracks your applications locally.

It is a hand-written tool-calling loop (no agent framework) over any
OpenAI-compatible LLM API, so it runs on free-tier providers and can be pointed at
other models by changing env vars.

Tools the model can call: `read_resume`, `fetch_url`, `save_job`, `list_jobs`, `update_job`.

## Setup
```bash
pip install -r requirements.txt
cp resume.sample.md resume.md   # then edit with your real resume
```

Set your provider (example: Google Gemini free tier via AI Studio key):
```bash
export LLM_API_KEY=your_key
export LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export LLM_MODEL=<a model name from the provider's docs>
```
Other providers: Groq `https://api.groq.com/openai/v1`, OpenRouter `https://openrouter.ai/api/v1`.
Free-tier limits and model names change, so check the provider's page. Use a model
that supports tool/function calling.

## Run
```bash
python agent.py                                   # interactive
python agent.py "Evaluate this job: https://..."  # one-shot
```

Example prompts:
- `Evaluate this job: <url or pasted description>`
- `Show my pipeline`
- `Mark job a1b2c3 as applied`

## How it works
`run_turn` sends the conversation plus tool schemas to the model. If the model
requests tools, they run locally and the results go back, until the model answers
without calling a tool (capped at 10 steps). Tool errors are returned to the model
as text so it can recover. Rewrites are constrained to your existing experience; the
agent is told never to invent anything. Applications are stored in `jobs.json`.

## Why I built it
(Fill in: your own reason, e.g. job-hunting was a manual copy-paste loop.)
