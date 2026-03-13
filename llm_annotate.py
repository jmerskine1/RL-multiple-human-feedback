#!/usr/bin/env python3
"""
llm_annotate.py — Use LLMs as synthetic trainers alongside human participants.

Each LLM is treated as one trainer in the multi-trainer VI framework, identical
in structure to a human participant.  The LLM is shown an ASCII grid of the
current Pac-Man state and asked to assign a numerical desirability score (0–1)
to each action — matching the ordinal-feedback format used in the framework.
Scores are stored as hp/hm arrays via the ordinal-feedback pathway in
_collect_feedback: scores > 0.5 accumulate in hp (positive), scores < 0.5
accumulate in hm (negative), with magnitude proportional to distance from 0.5.
Brain files are saved locally and/or uploaded to GCS, and load transparently
alongside human participant brains in offline_train.py.

The VI Ce-estimation in offline_train.py will automatically measure how
consistently each LLM's scored feedback aligns with the learned Q-table —
giving a principled, model-agnostic quality score for each annotator.

Supported LLMs
--------------
  claude     →  claude-opus-4-5           (Anthropic)
  claude-s   →  claude-sonnet-4-5         (Anthropic, cheaper)
  gpt4o      →  gpt-4o                    (OpenAI)
  gpt4o-mini →  gpt-4o-mini               (OpenAI, cheaper)

Output
------
  One brain file per LLM, saved locally and/or uploaded to GCS.
  Session names follow the convention  LLM_<name>  (e.g. LLM_claude, LLM_gpt4o)
  so they load transparently with:
      python offline_train.py --sessions P001 P002 LLM_claude LLM_gpt4o ...

Usage
-----
  # Annotate with Claude and GPT-4o, upload to GCS:
  python llm_annotate.py \\
      --llms claude gpt4o \\
      --bucket <your-gcs-bucket> \\
      --n-rollout-episodes 20

  # Annotate and save locally only (no GCS):
  python llm_annotate.py \\
      --llms claude gpt4o-mini \\
      --save-dir llm_brains/

  # Combine with human participants in offline training:
  python offline_train.py \\
      --bucket <your-gcs-bucket> \\
      --sessions P001 P002 LLM_claude LLM_gpt4o \\
      --n-episodes 1000

Authentication
--------------
  export ANTHROPIC_API_KEY=sk-ant-...
  export OPENAI_API_KEY=sk-...
  gcloud auth application-default login   # only needed if --bucket is set
"""

import os
import re
import json
import pickle
import time
import argparse
import numpy as np

from feedback import Feedback
from trainer import PacmanTrainer


# ── Rate-limit helpers ────────────────────────────────────────────────────────

def _is_rate_limit(exc) -> bool:
    """Return True if the exception looks like a 429 / quota-exhausted error."""
    msg = str(exc).lower()
    return any(k in msg for k in (
        "429", "resource_exhausted", "rate_limit", "ratelimit",
        "too many requests", "quota",
    ))


def _is_not_found(exc) -> bool:
    """Return True if the model doesn't exist (404 not_found_error)."""
    msg = str(exc).lower()
    return "not_found_error" in msg or ("404" in msg and "not found" in msg)


def _is_daily_quota(exc) -> bool:
    """
    Return True if the quota error is a *daily* (not per-minute) limit.

    Google embeds quota IDs like 'GenerateRequestsPerDayPerProjectPerModel'
    in the error body.  These reset at UTC midnight; retrying within the same
    session is pointless.

    OpenAI's billing-exhausted error has code 'insufficient_quota' — also
    unrecoverable without user action, so treat it the same way.
    """
    text = str(exc)
    return any(k in text for k in (
        "PerDay", "per_day", "PerMonth", "per_month",
        "insufficient_quota",  # OpenAI: billing exhausted
    ))


def _retry_delay_seconds(exc, default: float = 20.0) -> float:
    """
    Try to parse the suggested retry delay from a rate-limit exception.

    Google embeds  retryDelay: "20s"  in the error details.
    OpenAI/Anthropic embed  'retry after N'  in the message.
    Falls back to `default` if no value is found.
    """
    text = str(exc)
    # Google: retryDelay\nvalue: "20s"  or  retryDelay: 20s
    m = re.search(r'retryDelay[^0-9]*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # OpenAI / Anthropic: "Retry-After: 30" or "retry after 30"
    m = re.search(r'retry[^0-9]*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return default


# ── LLM model registry ────────────────────────────────────────────────────────
#
# Models are grouped into two tiers so you can run a fair comparison:
#
#   LARGE tier — flagship models, highest capability, higher cost
#     claude-large   →  claude-opus-4-6              (Anthropic)
#     gpt4o          →  gpt-4o                       (OpenAI, paid API)
#     gemini-large   →  gemini-2.5-pro               (Google, Gemini Pro)
#
#   SMALL tier — capable but cheap; comparable in size and cost across providers
#     claude-small   →  claude-haiku-4-5-20251001    (Anthropic, cheapest)
#     gpt4o-mini     →  gpt-4o-mini                  (OpenAI, ~$0.15/1M tokens)
#     gemini-small   →  gemini-2.0-flash             (Google, cheapest)
#
# Fair comparison options:
#   --tier small   →  use claude-small, gpt4o-mini, gemini-small
#   --tier large   →  use claude-large, gpt4o, gemini-large
#                     (requires paid OpenAI API; not accessible on free tier)
#
# For your setup (Claude Pro, Gemini Pro, free ChatGPT):
#   Recommended: --tier small
#   This uses claude-haiku, gpt-4o-mini, gemini-flash — all comparable in
#   capability and cost.  gpt-4o-mini is not "free" but costs ~$0.01 for the
#   entire annotation run (a few hundred states × ~200 tokens each).
#   Free ChatGPT web access does NOT provide API access; you need an API key
#   even on the free tier.  A new OpenAI account gets $5 of free API credits,
#   which is more than enough.

LLM_REGISTRY = {
    # ── Large / flagship ──────────────────────────────────────────────────────
    "claude-large":  {"provider": "anthropic", "model": "claude-opus-4-6",
                      "tier": "large"},
    "gpt4o":         {"provider": "openai",    "model": "gpt-4o",
                      "tier": "large"},
    "gemini-large":  {"provider": "google",    "model": "gemini-2.5-pro",
                      "tier": "large"},
    # ── Small / mini — recommended for fair cross-provider comparison ──────────
    "claude-small":  {"provider": "anthropic", "model": "claude-haiku-4-5-20251001",
                      "tier": "small"},
    "gpt4o-mini":    {"provider": "openai",    "model": "gpt-4o-mini",
                      "tier": "small"},
    "gemini-small":  {"provider": "google",    "model": "gemini-2.5-flash",
                      "tier": "small"},
}

# Shortcut: --tier small/large selects all three providers at that tier
TIER_SHORTCUTS = {
    "small": ["claude-small", "gpt4o-mini", "gemini-small"],
    "large": ["claude-large", "gpt4o",      "gemini-large"],
}


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Pac-Man player advising a reinforcement learning agent.
You will be shown the current game state as an ASCII grid. Your job is to
assign a desirability score between 0.0 and 1.0 to each of the four movement
actions, where 1.0 means "strongly prefer this action" and 0.0 means "strongly
avoid this action".  Scores near 0.5 indicate uncertainty or indifference.

Grid legend:
  #  = wall (cannot move through)
  P  = Pac-Man (the agent you are advising)
  G  = ghost   (avoid — touching the ghost ends the episode with a penalty)
  *  = pellet  (collect for a reward)
     = empty space

Actions:  n = move north (up)
          s = move south (down)
          e = move east  (right)
          w = move west  (left)

Note: moving into a wall leaves the agent stationary (score it low, not 0.5).

Respond with ONLY a JSON object — no prose, no markdown fences:
{
  "scores": {"n": 0.0, "s": 0.0, "e": 0.0, "w": 0.0},
  "reasoning": "one sentence"
}
All four actions must have a score. Scores must be in [0.0, 1.0].\
"""


def _make_user_prompt(display_lines: list) -> str:
    return "Current game state:\n\n" + "\n".join(display_lines) + \
           "\n\nScore each action from 0.0 (avoid) to 1.0 (prefer)."


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_scores(text: str, action_list: list) -> list:
    """
    Extract per-action desirability scores from an LLM response.

    Returns a list of floats in action_list order, each in [0.0, 1.0].
    Falls back to [0.5, 0.5, 0.5, 0.5] (neutral) if parsing fails, so no
    spurious hp/hm mass is injected on failure.
    """
    neutral = [0.5] * len(action_list)
    try:
        text = re.sub(r"```[a-z]*", "", text).strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        obj   = json.loads(text[start:end])
        raw   = obj.get("scores", {})
        scores = []
        for a in action_list:
            v = float(raw.get(a, 0.5))
            scores.append(float(np.clip(v, 0.0, 1.0)))
        return scores
    except Exception:
        return neutral   # fallback: neutral — no signal injected


# ── LLM query functions ───────────────────────────────────────────────────────
# Each function returns a list[float] of per-action scores in action_list order.
# Values are in [0.0, 1.0] — passed directly to Feedback as ordinal-feedback.

def _query_anthropic(client, model: str, display_lines: list,
                     action_list: list) -> list:
    msg = client.messages.create(
        model=model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": _make_user_prompt(display_lines)}],
    )
    return _parse_scores(msg.content[0].text, action_list)


def _query_openai(client, model: str, display_lines: list,
                  action_list: list) -> list:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _make_user_prompt(display_lines)},
        ],
    )
    return _parse_scores(resp.choices[0].message.content, action_list)


def _query_google(client, model: str, display_lines: list,
                  action_list: list) -> list:
    # google-genai: system prompt via config, user prompt as contents
    from google.genai import types as _gtypes
    resp = client.models.generate_content(
        model=model,
        config=_gtypes.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        contents=_make_user_prompt(display_lines),
    )
    return _parse_scores(resp.text, action_list)


# ── State collection ──────────────────────────────────────────────────────────

def collect_state_displays(env_size: str, n_episodes: int,
                           max_steps: int = 500) -> dict:
    """
    Run random-policy rollouts and collect a map of state_index → display lines.
    Using random=True starts ensures broad state coverage.
    Returns {state_int: [display_row_str, ...]}
    """
    trainer = PacmanTrainer(algID="tabQL_Cest_vi_t2", env_size=env_size)
    # Trigger Q/hp/hm initialisation
    trainer.agent.act(0, 0, 0.0, False, [[]], 0.5, update_Cest=False)

    state_displays = {}

    for ep in range(n_episodes):
        trainer.reset_episode(random=True)
        for _ in range(max_steps):
            state = trainer.ob
            if state not in state_displays:
                state_displays[state] = trainer.env.display()
            trainer.step(feedback=[[]], update_Cest=False)
            if trainer.done:
                break

    return state_displays


# ── Brain builder ─────────────────────────────────────────────────────────────

def build_llm_brain(llm_name: str, state_displays: dict, action_list: list,
                    env_size: str, query_fn, verbose: bool = True,
                    rate_limit: float = 0.0, max_retries: int = 3) -> dict:
    """
    Query the LLM for every collected state and accumulate hp/hm.

    Uses the ordinal-feedback pathway in _collect_feedback:
      Feedback(state=s, good_actions=[f_n, f_s, f_e, f_w], conf_good_actions=1.0)
    where each f_i is a float in [0, 1] returned by the LLM.

    Inside _collect_feedback, scores are interpreted as:
      score > 0.5  →  hp[n, s, i] += (score - 0.5) * 2   (positive evidence)
      score < 0.5  →  hm[n, s, i] += (0.5 - score) * 2   (negative evidence)
      score ≈ 0.5  →  no update                            (neutral / uncertain)

    Scores are min-max normalised per state before injection so that the full
    [0, 1] dynamic range is used, matching the ordinal-feedback generator in
    feedback.py (generate_single_feedback type='ordinal-feedback').

    The VI Ce-estimation will learn the reliability of each LLM's signal
    relative to the Q-table automatically.

    Parameters
    ----------
    rate_limit   : seconds to sleep between requests (0 = no throttle)
    max_retries  : number of retry attempts on rate-limit errors; each retry
                   sleeps for the delay suggested by the API (or 20 s default)

    Returns a brain dict in the same format as human participant brains,
    ready for upload_brain() or offline_train.py.
    """
    trainer = PacmanTrainer(algID="tabQL_Cest_vi_t2", env_size=env_size)
    trainer.agent.act(0, 0, 0.0, False, [[]], 0.5, update_Cest=False)

    n_states = len(state_displays)
    ok, failed = 0, 0
    abort = False        # set True → skip remaining states (daily quota or bad model)
    abort_reason = ""

    for i, (state_idx, display) in enumerate(state_displays.items()):
        if abort:
            failed += 1
            continue

        if verbose and (i % max(1, n_states // 10) == 0 or i == n_states - 1):
            print(f"  [{llm_name}]  {i+1:>4}/{n_states} states annotated "
                  f"({failed} failed)")

        # ── Retry loop (handles rate-limit 429s) ──────────────────────────────
        success = False
        for attempt in range(max_retries + 1):
            try:
                scores = query_fn(display, action_list)   # list[float]

                # Min-max normalise to [0, 1] so full dynamic range is used,
                # matching generate_single_feedback(type='ordinal-feedback').
                scores = np.array(scores, dtype=float)
                v_min, v_max = scores.min(), scores.max()
                if v_max > v_min:
                    scores = (scores - v_min) / (v_max - v_min)
                # else all identical → stays uniform, no net signal (neutral)

                fb = Feedback(state=state_idx,
                              good_actions=scores.tolist(),  # floats → ordinal path
                              conf_good_actions=1.0)
                trainer.agent._collect_feedback([[fb]])
                ok += 1
                success = True
                break   # success — exit retry loop

            except Exception as e:
                if _is_not_found(e):
                    # Model doesn't exist — no point trying any more states
                    print(f"\n  [{llm_name}]  *** Model not found: {e}")
                    print(f"  Check the model name in LLM_REGISTRY and update it.")
                    print(f"  Current model: {llm_name!r}")
                    abort = True
                    break

                elif _is_rate_limit(e) and _is_daily_quota(e):
                    # Daily cap exhausted — no point retrying within this session
                    print(f"\n  [{llm_name}]  *** Quota exhausted (unrecoverable) ***")
                    if "insufficient_quota" in str(e):
                        print(f"  OpenAI account has no remaining credits.")
                        print(f"  Add billing at https://platform.openai.com/account/billing")
                    else:
                        print(f"  The free tier allows a fixed number of requests per day (UTC).")
                        print(f"  Options:")
                        print(f"    1. Wait until tomorrow (UTC midnight) and rerun.")
                        print(f"    2. Enable billing to lift the daily cap.")
                    print(f"  Skipping remaining {n_states - i} states.\n")
                    abort = True
                    break   # break inner loop; outer loop will skip via flag

                elif _is_rate_limit(e) and attempt < max_retries:
                    # Per-minute rate limit — sleep for the suggested delay, then retry
                    delay = max(_retry_delay_seconds(e), 1.0)  # never sleep < 1 s
                    print(f"  [{llm_name}]  rate-limited — "
                          f"sleeping {delay:.0f}s then retrying "
                          f"(attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)

                elif attempt < max_retries:
                    # Transient / unknown error — short exponential backoff
                    backoff = 2 ** attempt
                    print(f"  [{llm_name}]  error (attempt {attempt + 1}/"
                          f"{max_retries + 1}), retrying in {backoff}s: {e}")
                    time.sleep(backoff)

                else:
                    print(f"  [{llm_name}]  state {state_idx} failed after "
                          f"{max_retries + 1} attempt(s): {e}")

        if not success:
            failed += 1

        # ── Inter-request throttle (skip after last state) ────────────────────
        if rate_limit > 0 and i < n_states - 1 and not abort:
            time.sleep(rate_limit)

    total_signal = float(np.sum(trainer.agent.hp) + np.sum(trainer.agent.hm))
    print(f"  [{llm_name}]  done — {ok}/{n_states} states, "
          f"total feedback mass: {total_signal:.1f}")

    return {
        "Q":                      trainer.agent.Q,
        "hp":                     trainer.agent.hp,
        "hm":                     trainer.agent.hm,
        "Ce":                     trainer.agent.Ce,
        "sum_of_right_feedback":  trainer.agent.sum_of_right_feedback,
        "sum_of_wrong_feedback":  trainer.agent.sum_of_wrong_feedback,
        "Nsa":                    trainer.agent.Nsa,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Annotate Pac-Man states with LLMs and save as brain files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tier shortcuts (recommended for fair cross-provider comparison):
  --tier small   claude-small + gpt4o-mini + gemini-small   (cheap, balanced)
  --tier large   claude-large + gpt4o     + gemini-large    (flagship, needs paid OpenAI)

Your setup (Claude Pro, Gemini Pro, free ChatGPT):
  Use --tier small.  gpt4o-mini needs an API key but costs ~$0.01 for a full
  annotation run — new OpenAI accounts receive $5 of free API credits.
        """,
    )
    p.add_argument("--llms", nargs="+", default=None,
                   choices=list(LLM_REGISTRY.keys()), metavar="LLM",
                   help="Specific LLMs to use. Mutually exclusive with --tier.")
    p.add_argument("--tier", default=None, choices=list(TIER_SHORTCUTS.keys()),
                   help="Select all providers at this capability tier. "
                        "Mutually exclusive with --llms.")
    p.add_argument("--env-size", default="small",
                   choices=["small", "medium", "medium_sparse"])
    p.add_argument("--n-rollout-episodes", type=int, default=5,
                   help="Episodes to run for state collection (default: 5). "
                        "More episodes → broader state coverage.")
    p.add_argument("--max-steps", type=int, default=500,
                   help="Max steps per rollout episode (default: 500)")
    p.add_argument("--bucket", default=None,
                   help="GCS bucket to upload brain files to. "
                        "If omitted, brains are saved locally only.")
    p.add_argument("--save-dir", default="llm_brains",
                   help="Local directory to save brain files (default: llm_brains/)")
    p.add_argument("--rate-limit", type=float, default=4.0, metavar="SECS",
                   help="Seconds to wait between requests (default: 4.0). "
                        "Gemini free tier allows ~15 RPM, so 4 s keeps you "
                        "safely under quota.  Set to 0 for paid/high-quota tiers.")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max retry attempts per state on rate-limit errors "
                        "(default: 3).  Each retry waits for the delay "
                        "suggested by the API (typically 20 s for Gemini).")
    p.add_argument("--dry-run", action="store_true",
                   help="Collect states and print a sample prompt, "
                        "but do not make any LLM API calls.")
    args = p.parse_args()

    # Resolve --tier / --llms
    if args.tier and args.llms:
        p.error("--tier and --llms are mutually exclusive.")
    if args.tier:
        llm_names = TIER_SHORTCUTS[args.tier]
        print(f"  Tier '{args.tier}' selected: {llm_names}")
    elif args.llms:
        llm_names = args.llms
    else:
        p.error("Provide either --llms or --tier.")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. Initialise API clients ──────────────────────────────────────────────
    anthropic_client, openai_client, google_clients = None, None, {}
    providers_needed = {LLM_REGISTRY[name]["provider"] for name in llm_names}

    if not args.dry_run:
        from secrets_loader import secret as _secret
        if "anthropic" in providers_needed:
            api_key = _secret("anthropic_api_key", env_var="ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "Anthropic API key not found. Set 'anthropic_api_key' in secrets.json "
                    "or: export ANTHROPIC_API_KEY=sk-ant-...")
            import anthropic as _anthropic
            anthropic_client = _anthropic.Anthropic(api_key=api_key)
            print("  Anthropic client initialised.")

        if "openai" in providers_needed:
            api_key = _secret("openai_api_key", env_var="OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OpenAI API key not found. Set 'openai_api_key' in secrets.json "
                    "or: export OPENAI_API_KEY=sk-...")
            import openai as _openai
            openai_client = _openai.OpenAI(api_key=api_key)
            print("  OpenAI client initialised.")

        if "google" in providers_needed:
            api_key = _secret("google_api_key", env_var="GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "Google API key not found. Set 'google_api_key' in secrets.json "
                    "or: export GOOGLE_API_KEY=AIza...")
            try:
                from google import genai as _genai
                # Single client shared across all google models
                google_clients["_client"] = _genai.Client(api_key=api_key)
                print("  Google Gemini client initialised.")
            except ImportError:
                raise ImportError(
                    "google-genai not installed. Run: pip install google-genai")

    # ── 2. Collect states via rollout ──────────────────────────────────────────
    print(f"\n── Collecting states ({args.n_rollout_episodes} episodes, "
          f"env={args.env_size}) ──")
    state_displays = collect_state_displays(
        args.env_size, args.n_rollout_episodes, args.max_steps)
    action_list = ["n", "s", "e", "w"]
    print(f"  {len(state_displays)} unique states collected.\n")

    if args.dry_run:
        # Print a sample prompt so the user can verify formatting before spending credits
        sample_state = next(iter(state_displays))
        print("── Sample prompt (first collected state) ──")
        print(SYSTEM_PROMPT)
        print()
        print(_make_user_prompt(state_displays[sample_state]))
        print(f"\n── Expected response format ──")
        print('{"scores": {"n": 0.8, "s": 0.1, "e": 0.6, "w": 0.05},')
        print(' "reasoning": "move toward nearest pellet, away from ghost"}')
        print(f"\n(dry run — {len(state_displays)} states would be scored "
              f"per LLM, no API calls made)")
        return

    # ── 3. Annotate with each LLM ──────────────────────────────────────────────
    for llm_name in llm_names:
        cfg      = LLM_REGISTRY[llm_name]
        provider = cfg["provider"]
        model    = cfg["model"]
        tier     = cfg["tier"]
        session  = f"LLM_{llm_name}"

        print(f"── Annotating with {llm_name} [{tier}] ({model}) "
              f"→ session: {session} ──")

        if provider == "anthropic":
            query_fn = lambda disp, acts, _m=model: \
                _query_anthropic(anthropic_client, _m, disp, acts)
        elif provider == "openai":
            query_fn = lambda disp, acts, _m=model: \
                _query_openai(openai_client, _m, disp, acts)
        else:  # google
            gc = google_clients["_client"]
            query_fn = lambda disp, acts, _gc=gc, _m=model: \
                _query_google(_gc, _m, disp, acts)

        brain = build_llm_brain(
            llm_name, state_displays, action_list,
            env_size=args.env_size,
            query_fn=query_fn,
            rate_limit=args.rate_limit,
            max_retries=args.max_retries,
        )

        # ── Save locally ───────────────────────────────────────────────────────
        local_path = os.path.join(args.save_dir, f"{session}_brain.pkl")
        with open(local_path, "wb") as f:
            pickle.dump(brain, f)
        print(f"  Saved locally → {local_path}")

        # ── Upload to GCS ──────────────────────────────────────────────────────
        if args.bucket:
            from gcs_utils import upload_brain
            upload_brain(session, brain, args.bucket)
            print(f"  Uploaded → gs://{args.bucket}/brains/{session}/brain.pkl")

        print()

    # ── 4. Summary ────────────────────────────────────────────────────────────
    session_names = [f"LLM_{n}" for n in llm_names]
    print("── Done ──")
    print(f"  LLM sessions: {session_names}")
    if args.bucket:
        all_sessions = " ".join(session_names)
        print(f"\n  Run offline training with:")
        print(f"    python offline_train.py \\")
        print(f"        --bucket {args.bucket} \\")
        print(f"        --sessions P001 P002 {all_sessions} \\")
        print(f"        --n-episodes 1000")
    else:
        print(f"\n  Brain files saved to: {args.save_dir}/")
        print(f"  Upload manually with:")
        for name in llm_names:
            session = f"LLM_{name}"
            print(f"    gsutil cp {args.save_dir}/{session}_brain.pkl "
                  f"gs://<bucket>/brains/{session}/brain.pkl")


if __name__ == "__main__":
    main()
