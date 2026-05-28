"""
Flask server for human experiment baseline.

Each participant completes all task types (one variant each), with gen tasks
within each type played sequentially. Task-type order is randomised per
participant. Results are saved to results/human/<participant_id>_<session_id>.json
and updated after every completed episode.

Usage (from repo root, inside tmux):
    python -m human_experiment.server --port 5050

    # Subset of task types:
    python -m human_experiment.server --task_types additive compositional --port 5050

    # Nonce surface names:
    python -m human_experiment.server --nonce --port 5050
"""

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, jsonify, request, send_from_directory

from env.env import AdventureEnv
from pipeline.prompts import GENERALIZATION_BASE_PROMPT
from tree_management.demo_generator import generate_demos, generate_mixed_demos
from tree_management.elements import fill_elements, load_lexicons
from tree_management.generator import generate_tasks
from tree_management.registry import get_task

ALL_TASK_TYPES = [
    "additive", "compositional", "conditional", "override",
    "proc_add", "proc_comp", "proc_cond", "proc_over",
]
RULES_DIR = os.path.join(os.path.dirname(__file__), "..", "tree_management", "rules")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "human")


def construct_demo_context(demos, include_world=True):
    if not demos:
        return ""
    lines = []
    if include_world:
        world = demos[0].metadata.get("world_listing", "")
        if world:
            lines.append(world)
            lines.append("")
    lines.append("[Start of Demonstration Episodes]")
    for demo in demos:
        lines.append(demo.format(show_world=False))
        lines.append("")
    lines.append("[End of Demonstration Episodes]")
    return "\n".join(lines)

app = Flask(__name__, static_folder=os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Global task store (populated once at startup, read-only after)
# ---------------------------------------------------------------------------

# task_type → {demo_context, gen_tasks, source_tasks, initial_currency}
_task_store: Dict[str, Dict] = {}

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_sessions: Dict[str, Dict] = {}
_sessions_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def _generate_one_type(
    task_type: str,
    use_nonce: bool,
    seed: int,
    initial_currency: int,
    distractor_rules_path: Optional[str],
    num_distractor_samples: int,
) -> Dict:
    import random as _random

    rules_path = os.path.join(RULES_DIR, f"{task_type}.json")
    with open(rules_path) as f:
        rule_spec = json.load(f)

    spec = get_task(task_type)
    sem_lex, nonce_lex = load_lexicons(rule_spec.get("lexicon", "default"))
    elements = fill_elements(rule_spec, sem_lex, nonce_lex, seed=seed, split_spec=spec.split)

    source_tasks = spec.gen_fn(elements, split="source", use_nonce=use_nonce)
    gen_tasks    = spec.gen_fn(elements, split="gen",    use_nonce=use_nonce)

    demo_source = list(source_tasks)
    _random.Random(seed).shuffle(demo_source)

    dist_path = distractor_rules_path or spec.distractor_rules
    if dist_path:
        from tree_management.generator import get_template
        with open(dist_path) as f:
            dist_spec = json.load(f)
        dist_sem_lex, dist_nonce_lex = load_lexicons(dist_spec.get("lexicon", "default"))
        dist_elems = fill_elements(dist_spec, dist_sem_lex, dist_nonce_lex, seed=seed)
        dist_tmpl  = get_template(spec.distractor_template or spec.template_name)
        all_dist   = generate_tasks(
            dist_elems, process=None, split=None,
            use_nonce=use_nonce, task_type="distractor",
            template_name=dist_tmpl.name, seed=seed,
        )
        distractor_tasks = (
            all_dist if num_distractor_samples >= len(all_dist)
            else _random.Random(seed).sample(all_dist, num_distractor_samples)
        )
        demos = generate_mixed_demos(
            property_tasks=demo_source,
            distractor_tasks=distractor_tasks,
            gen_tasks=gen_tasks,
            initial_currency=initial_currency,
            seed=seed,
        )
    else:
        demos = generate_demos(demo_source, initial_currency=initial_currency, seed=seed)

    demo_context = construct_demo_context(demos)
    print(f"  [{task_type}] {len(gen_tasks)} gen tasks, demo context {len(demo_context)} chars")

    return {
        "demo_context":     demo_context,
        "gen_tasks":        gen_tasks,
        "source_tasks":     list(source_tasks),
        "initial_currency": initial_currency,
    }


def build_task_store(
    task_types: List[str],
    use_nonce: bool = False,
    seed: int = 0,
    initial_currency: int = 500,
    distractor_rules_path: Optional[str] = None,
    num_distractor_samples: int = 4,
) -> Dict:
    store = {}
    print(f"[server] Generating tasks for: {task_types}")
    for tt in task_types:
        store[tt] = _generate_one_type(
            tt, use_nonce, seed, initial_currency,
            distractor_rules_path, num_distractor_samples,
        )
    print(f"[server] All tasks ready.")
    return store


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _make_env(task_type: str, gen_idx: int) -> tuple:
    entry            = _task_store[task_type]
    task             = entry["gen_tasks"][gen_idx]
    source_tasks     = entry["source_tasks"]
    initial_currency = entry["initial_currency"]

    all_trees = [(task.tree, task.tree.root_id)]
    for st in source_tasks:
        all_trees.append((st.tree, st.tree.root_id))

    env = AdventureEnv(
        trees=all_trees,
        initial_currency=initial_currency,
        initial_location="GameStart",
    )
    rule_seed = int(hashlib.md5(task.tree.root_id.encode()).hexdigest()[:8], 16) % (2**31)
    initial_obs = env.reset(
        tree_index=0,
        initial_currency=initial_currency,
        initial_location="GameStart",
        seed=rule_seed,
        rules_to_skip=task.rules_to_skip,
        task_label="Your task",
    )
    return env, initial_obs, task


def _episode_context(task_type: str, gen_idx: int) -> str:
    """Full prompt string (base + demo) for a given task type."""
    demo_context = _task_store[task_type]["demo_context"]
    prompt = GENERALIZATION_BASE_PROMPT
    if demo_context:
        prompt = prompt + "\n\n" + demo_context
    return prompt


def _progress(sess: Dict) -> Dict:
    type_idx      = sess["type_idx"]
    task_type     = sess["task_order"][type_idx]
    num_gen_tasks = len(_task_store[task_type]["gen_tasks"])
    return {
        "type_idx":      type_idx,
        "num_types":     len(sess["task_order"]),
        "gen_idx":       sess["gen_idx"],
        "num_gen_tasks": num_gen_tasks,
        "task_type":     task_type,
    }


def _save(sess: Dict, session_id: str) -> None:
    """Single file per participant: <pid>_<session_id>.json.
    Contains both episode results (for analysis) and resume state (task_order, indices).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pid  = sess["participant_id"].replace("/", "_").replace(" ", "_")
    path = os.path.join(RESULTS_DIR, f"{pid}_{session_id}.json")
    out  = {
        "session_id":      session_id,
        "participant_id":  sess["participant_id"],
        "task_order":      sess["task_order"],
        "type_idx":        sess["type_idx"],
        "gen_idx":         sess["gen_idx"],
        "start_time":      sess["start_time"],
        "last_updated":    datetime.now(timezone.utc).isoformat(),
        "all_done":        sess.get("all_done", False),
        "episode_results": sess["episode_results"],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def _finalise_episode(sess: Dict, session_id: str) -> None:
    """Append current episode to results and persist."""
    task_type = sess["task_order"][sess["type_idx"]]
    task      = sess["task"]
    env       = sess["env"]

    sess["episode_results"].append({
        "task_type":        task_type,
        "type_idx":         sess["type_idx"],
        "gen_idx":          sess["gen_idx"],
        "root_id":          task.tree.root_id,
        "success":          env.done,
        "num_runs":         sess["step"],
        "history":          list(sess["history"]),
        "reference_solution": task.tree.get_solution(),
        "reference_length": len(task.tree.get_solution()),
        "currency_remaining": env.currency,
        "episode_start_time": sess["episode_start_time"],
        "episode_end_time":   datetime.now(timezone.utc).isoformat(),
    })
    _save(sess, session_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "interface.html")


@app.route("/api/start", methods=["POST"])
def start_session():
    """
    Create a new participant session.
    Returns the first episode's prompt, initial_obs, and progress info.
    """
    data           = request.get_json(silent=True) or {}
    participant_id = data.get("participant_id", "anonymous").strip() or "anonymous"

    task_order = list(_task_store.keys())
    random.shuffle(task_order)

    session_id = str(uuid.uuid4())
    now        = datetime.now(timezone.utc).isoformat()

    env, initial_obs, task = _make_env(task_order[0], 0)

    sess = {
        "participant_id":   participant_id,
        "task_order":       task_order,
        "type_idx":         0,
        "gen_idx":          0,
        "env":              env,
        "task":             task,
        "step":             0,
        "history":          [],
        "episode_results":  [],
        "start_time":       now,
        "episode_start_time": now,
        "all_done":         False,
    }

    with _sessions_lock:
        _sessions[session_id] = sess

    prog = _progress(sess)
    task_type = prog["task_type"]
    return jsonify({
        "session_id":    session_id,
        "base_prompt":   GENERALIZATION_BASE_PROMPT,
        "demo_context":  _task_store[task_type]["demo_context"],
        "initial_obs":   initial_obs,
        **prog,
        "reference_length": len(task.tree.get_solution()),
    })


@app.route("/api/resume", methods=["POST"])
def resume_session():
    """
    Resume a session from a saved session code.
    The current in-progress episode restarts from the beginning.
    """
    data         = request.get_json(silent=True) or {}
    session_code = data.get("session_code", "").strip()

    # Search for the single participant file matching this session code
    os.makedirs(RESULTS_DIR, exist_ok=True)
    matches = [f for f in os.listdir(RESULTS_DIR) if f.endswith(f"_{session_code}.json")]
    if not matches:
        return jsonify({"error": "Session code not found. Check the code and try again."}), 404
    path = os.path.join(RESULTS_DIR, matches[0])

    with open(path) as f:
        saved = json.load(f)

    if saved.get("all_done"):
        return jsonify({"error": "This session is already complete."}), 400

    session_id      = saved["session_id"]
    task_order      = saved["task_order"]
    type_idx        = saved["type_idx"]
    gen_idx         = saved["gen_idx"]
    episode_results = saved["episode_results"]
    task_type       = task_order[type_idx]

    env, initial_obs, task = _make_env(task_type, gen_idx)
    now = datetime.now(timezone.utc).isoformat()

    sess = {
        "participant_id":     saved["participant_id"],
        "task_order":         task_order,
        "type_idx":           type_idx,
        "gen_idx":            gen_idx,
        "env":                env,
        "task":               task,
        "step":               0,
        "history":            [],
        "episode_results":    episode_results,
        "start_time":         saved["start_time"],
        "episode_start_time": now,
        "all_done":           False,
    }

    with _sessions_lock:
        _sessions[session_id] = sess

    prog = _progress(sess)
    return jsonify({
        "session_id":         session_id,
        "base_prompt":        GENERALIZATION_BASE_PROMPT,
        "demo_context":       _task_store[task_type]["demo_context"],
        "initial_obs":        initial_obs,
        **prog,
        "reference_length":   len(task.tree.get_solution()),
        "resumed":            True,
        "completed_episodes": len(episode_results),
    })


@app.route("/api/action", methods=["POST"])
def take_action():
    data       = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")
    action     = data.get("action", "").strip().lower()
    argument   = data.get("argument", "").strip()

    with _sessions_lock:
        sess = _sessions.get(session_id)
    if sess is None:
        return jsonify({"error": "Unknown session. Please start a new session."}), 400
    if sess.get("all_done"):
        return jsonify({"error": "All tasks complete."}), 400

    env = sess["env"]
    if env.done:
        return jsonify({"error": "Episode already finished — call /api/next_episode."}), 400

    sess["step"] += 1
    step_num = sess["step"]

    full_action, obs_obj, _ = env.step(action, argument)
    observation    = obs_obj.message
    done           = obs_obj.done
    action_success = obs_obj.success

    sess["history"].append({
        "step":           step_num,
        "action":         full_action,
        "observation":    observation,
        "action_success": action_success,
        "done":           done,
    })

    resp = {
        "step":            step_num,
        "full_action":     full_action,
        "observation":     observation,
        "action_success":  action_success,
        "done":            done,
        "episode_success": env.done,
        "location":        env.current_location,
        "inventory":       dict(env.inventory),
        "currency":        env.currency,
    }

    if done:
        _finalise_episode(sess, session_id)

    return jsonify(resp)


@app.route("/api/next_episode", methods=["POST"])
def next_episode():
    """
    Advance to the next gen task (within type) or next task type.
    If the current episode was not finished, it is recorded as a give-up.
    Returns the next episode's context, or {all_done: true}.
    """
    data       = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")

    with _sessions_lock:
        sess = _sessions.get(session_id)
    if sess is None:
        return jsonify({"error": "Unknown session."}), 400

    # Finalise unfinished episode as give-up
    if not sess["env"].done:
        sess["history"].append({
            "step": sess["step"], "action": "GAVE_UP",
            "observation": "", "action_success": False, "done": True,
        })
        _finalise_episode(sess, session_id)

    # Advance
    task_type     = sess["task_order"][sess["type_idx"]]
    num_gen_tasks = len(_task_store[task_type]["gen_tasks"])
    new_task_type = False

    if sess["gen_idx"] + 1 < num_gen_tasks:
        sess["gen_idx"] += 1
    else:
        sess["type_idx"] += 1
        sess["gen_idx"]   = 0
        new_task_type     = True

    if sess["type_idx"] >= len(sess["task_order"]):
        sess["all_done"] = True
        _save(sess, session_id)
        return jsonify({"all_done": True})

    # Set up next episode
    next_type = sess["task_order"][sess["type_idx"]]
    env, initial_obs, task = _make_env(next_type, sess["gen_idx"])
    now = datetime.now(timezone.utc).isoformat()
    sess.update({
        "env":                env,
        "task":               task,
        "step":               0,
        "history":            [],
        "episode_start_time": now,
    })

    prog = _progress(sess)
    resp = {
        "all_done":    False,
        "new_task_type": new_task_type,
        "initial_obs": initial_obs,
        **prog,
        "reference_length": len(task.tree.get_solution()),
    }
    if new_task_type:
        resp["demo_context"] = _task_store[next_type]["demo_context"]

    return jsonify(resp)


@app.route("/api/give_up", methods=["POST"])
def give_up():
    """Give up the current episode (recorded as failure), then advance."""
    data       = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "")

    with _sessions_lock:
        sess = _sessions.get(session_id)
    if sess is None:
        return jsonify({"error": "Unknown session."}), 400

    sess["history"].append({
        "step": sess["step"], "action": "GAVE_UP",
        "observation": "", "action_success": False, "done": True,
    })
    _finalise_episode(sess, session_id)
    # Force env.done so next_episode knows not to double-record
    sess["env"].done = True

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Human experiment server")
    p.add_argument("--task_types", nargs="+", default=ALL_TASK_TYPES,
                   choices=ALL_TASK_TYPES + ["all"],
                   help="Task types to include. Default: all 8.")
    p.add_argument("--nonce", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial_currency", type=int, default=500)
    p.add_argument("--distractor_rules", default=None)
    p.add_argument("--num_distractor_samples", type=int, default=4)
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--host", default="0.0.0.0")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    task_types = ALL_TASK_TYPES if "all" in args.task_types else args.task_types

    _task_store.update(build_task_store(
        task_types=task_types,
        use_nonce=args.nonce,
        seed=args.seed,
        initial_currency=args.initial_currency,
        distractor_rules_path=args.distractor_rules,
        num_distractor_samples=args.num_distractor_samples,
    ))

    print(f"[server] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
