"""Two-phase generalization pipeline: Phase 1 (demo only) and Phase 2 (demo + teaching)."""

import os
import json
import random as _random
import statistics
from typing import Any, Optional, List, Dict, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from herosjourney.core.generator import GeneratedTask
from herosjourney.core.demo_generator import (
    generate_demos, generate_mixed_demos, build_world_listing,
)
from herosjourney.runner.adventure_episode import (
    run_single_episode,
    format_episode_result,
    construct_demo_context,
)


RESULTS_DIR = "./results"
VARIANTS_DIR = "./world_info/variants"


# ---------------------------------------------------------------------------
# Teaching helpers
# ---------------------------------------------------------------------------

def _episode_metrics(result) -> Dict:
    """Minimal metrics dict for teacher curation from an EpisodeResult."""
    return {
        "success":                         result.success,
        "completion_rate":                 result.completion_rate,
        "num_runs":                        result.num_runs,
        "repetition_fraction":             0.0,
        "generalization_mentions_total":   0,
        "generalization_mentions_by_term": {},
    }


def save_phase1_1_results(results: Dict, save_name: str) -> str:
    """Save Phase 1.1 curation results (teaching message + history) to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = f"{RESULTS_DIR}/phase_1_1_{save_name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Phase 1.1 results saved to: {path}")
    return path


def load_phase1_1_results(save_name: str, custom_path: Optional[str] = None) -> Optional[Dict]:
    path = custom_path or f"{RESULTS_DIR}/phase_1_1_{save_name}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_phase_results(results: Dict, phase_name: str, save_name: str) -> str:
    """Save phase results to JSON (strips _ prefixed internal fields)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = f"{RESULTS_DIR}/phase_{phase_name}_{save_name}.json"
    clean = {k: v for k, v in results.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(clean, f, indent=4)
    print(f"Phase {phase_name} results saved to: {path}")
    return path


def load_phase_results(
    phase_name: str,
    save_name: str,
    custom_path: Optional[str] = None,
) -> Optional[Dict]:
    path = custom_path or f"{RESULTS_DIR}/phase_{phase_name}_{save_name}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _cost_for_model(model_path: str) -> Tuple[float, float]:
    if model_path == "gemini":
        return 0.1, 0.4
    if model_path == "claude4.5":
        return 3.0, 15.0
    return 0.0, 0.0


def _format_batch_results(
    raw_results: Dict,
    key_prefix: str,
    model_path: str = "",
) -> Tuple[Dict, int, Dict]:
    """Format raw episode results into (episodes_dict, success_count, token_stats)."""
    episodes: Dict = {}
    success_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for idx in sorted(raw_results.keys()):
        result = raw_results[idx]
        total_prompt_tokens     += result.prompt_tokens
        total_completion_tokens += result.completion_tokens
        formatted = format_episode_result(result)
        episodes[f"{key_prefix}_{idx}"] = formatted
        if formatted["success"]:
            success_count += 1

    in_cost, out_cost = _cost_for_model(model_path)
    total_tokens = total_prompt_tokens + total_completion_tokens
    estimated_cost = (
        (total_prompt_tokens / 1_000_000) * in_cost
        + (total_completion_tokens / 1_000_000) * out_cost
    )
    token_stats = {
        "total_prompt_tokens":     total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens":            total_tokens,
        "estimated_cost_usd":      estimated_cost,
    }
    return episodes, success_count, token_stats


def _phase_summary(episodes: Dict, success_count: int, token_stats: Dict, **extra) -> Dict:
    n = len(episodes)
    result = {
        "episodes":      episodes,
        "num_episodes":  n,
        "successes":     success_count,
        "failures":      n - success_count,
        "success_rate":  success_count / n if n else 0.0,
        "completion_rate": (
            sum(v["completion_rate"] for v in episodes.values()) / n if n else 0.0
        ),
        **extra,
    }
    if token_stats.get("total_tokens", 0) > 0:
        result["_token_usage"] = token_stats
    return result


def run_episodes_batch(
    gen_tasks: List[GeneratedTask],
    demo_context: str,
    max_runs: Optional[int],
    verbose: bool,
    truncate_window: Optional[int],
    model_path: str,
    initial_currency: int,
    num_workers: int,
    converter_model: str = "small",
    teaching_message: Union[str, List[str]] = "",
    phase_label: str = "",
    num_tries: int = 2,
    variant_seed: int = 0,
    source_tasks: Optional[List[GeneratedTask]] = None,
    episode_mode: str = "standard",
    no_reasoning: bool = False,
) -> Dict:
    """Run a batch of gen episodes, returning {episode_idx: raw_result}."""
    if isinstance(teaching_message, list):
        msgs = teaching_message + [""] * max(0, len(gen_tasks) - len(teaching_message))
    else:
        msgs = [teaching_message] * len(gen_tasks)

    episode_tasks = [(i, gen_tasks[i], msgs[i]) for i in range(len(gen_tasks))]
    raw_results: Dict = {}

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map = {
                executor.submit(
                    run_single_episode,
                    episode_idx=i,
                    task=task,
                    demo_context=demo_context,
                    max_runs=max_runs,
                    verbose=verbose,
                    truncate_window=truncate_window,
                    model_path=model_path,
                    initial_currency=initial_currency,
                    converter_model=converter_model,
                    teaching_message=msg,
                    num_tries=num_tries,
                    episode_label=f"v{variant_seed}_ep{i}",
                    source_tasks=source_tasks,
                    episode_mode=episode_mode,
                    no_reasoning=no_reasoning,
                ): i
                for i, task, msg in episode_tasks
            }
            with tqdm(total=len(episode_tasks), desc=phase_label or "Episodes") as pbar:
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        raw_results[idx] = future.result()
                    except Exception as e:
                        print(f"\nEpisode {idx} failed: {e}")
                    pbar.update(1)
    else:
        for i, task, msg in tqdm(episode_tasks, desc=phase_label or "Episodes"):
            try:
                raw_results[i] = run_single_episode(
                    episode_idx=i,
                    task=task,
                    demo_context=demo_context,
                    max_runs=max_runs,
                    verbose=verbose,
                    truncate_window=truncate_window,
                    model_path=model_path,
                    initial_currency=initial_currency,
                    converter_model=converter_model,
                    teaching_message=msg,
                    num_tries=num_tries,
                    episode_label=f"v{variant_seed}_ep{i}",
                    source_tasks=source_tasks,
                    episode_mode=episode_mode,
                    no_reasoning=no_reasoning,
                )
            except Exception as e:
                print(f"\nEpisode {i} failed: {e}")

    return raw_results


# ---------------------------------------------------------------------------
# Variant aggregation
# ---------------------------------------------------------------------------

def _aggregate_variant_results(
    variant_results: List[Dict],
    num_variants: int,
    demo_repeats: int,
) -> Dict:
    """Combine per-variant phase summaries into one aggregated results dict."""
    rates      = [v["success_rate"] for v in variant_results]
    total_succ = sum(v["successes"]    for v in variant_results)
    total_n    = sum(v["num_episodes"] for v in variant_results)

    return {
        "num_variants":              num_variants,
        "demo_repeats":              demo_repeats,
        "variants":                  variant_results,
        "num_episodes":              total_n,
        "successes":                 total_succ,
        "failures":                  total_n - total_succ,
        "success_rate":              total_succ / total_n if total_n else 0.0,
        "mean_variant_success_rate": statistics.mean(rates) if rates else 0.0,
        "std_variant_success_rate":  statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "completion_rate":           (
            sum(v["completion_rate"] for v in variant_results) / len(variant_results)
            if variant_results else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Two-phase pipeline
# ---------------------------------------------------------------------------

def run_two_phase_pipeline(
    task_type: str,
    elements_path: str,
    model_path: str,
    # Demo options
    distractor_rules_path: Optional[str] = None,
    num_distractor_samples: int = 4,
    use_nonce: bool = False,
    # Episode options
    max_runs: Optional[int] = None,
    num_tries: Union[int, str] = 2,
    verbose: bool = False,
    truncate_window: Optional[int] = None,
    initial_currency: int = 500,
    num_workers: int = 1,
    converter_model: str = "small",
    # Phase 2 teaching
    teaching_strategy: str = "none",
    teacher_model_path: Optional[str] = None,
    # Variant scaling
    num_variants: int = 1,
    demo_repeats: int = 1,
    variants_dir: Optional[str] = None,
    # Coverage sweep
    n_source_demos: Optional[int] = None,
    n_gen_tasks: Optional[int] = None,
    # Persistence
    save_name: Optional[str] = None,
    skip_phase1: bool = False,
    skip_phase2: bool = False,
    phase1_results_file: Optional[str] = None,
    phase1_1_results_file: Optional[str] = None,
    phase2_results_file: Optional[str] = None,
    no_reasoning: bool = False,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Run Phase 1 (demo only) and Phase 2 (demo + teaching) over num_variants seeds."""
    from herosjourney.core.registry import get_task
    from herosjourney.core.generator import get_template, generate_tasks

    spec   = get_task(task_type)
    gen_fn = spec.gen_fn

    # Resolve "max" → registered per-task upper bound
    if num_tries == "max":
        if spec.max_tries is None:
            raise ValueError(
                f"num_tries='max' requested for task '{task_type}' but no max_tries is "
                "registered. Set max_tries in register_task() for this task."
            )
        num_tries = spec.max_tries
        print(f"[pipeline] num_tries='max' resolved to {num_tries} for task '{task_type}'")

    # Fall back to registry distractor config if not passed explicitly
    if distractor_rules_path is None:
        distractor_rules_path = spec.distractor_rules

    # --- Save name ---
    if save_name is None:
        model_stem = os.path.basename(model_path.rstrip("/")).replace(" ", "_")[:20]
        save_name = f"{task_type}_{model_stem}"

    # --- Check for existing saved results ---
    phase1_saved   = load_phase_results("1", save_name, phase1_results_file)
    phase1_1_saved = load_phase1_1_results(save_name, phase1_1_results_file)
    phase2_saved   = load_phase_results("2", save_name, phase2_results_file)

    if skip_phase1 and phase1_saved is None:
        print("ERROR: --skip_phase1 set but no saved Phase 1 results found.")
        return None, None

    run_p1 = not skip_phase1 and phase1_saved is None
    run_p2 = not skip_phase2 and phase2_saved is None

    if not run_p1 and not run_p2:
        # Both phases loaded from disk — just print summary
        for label, res in [("Phase 1 (demo only)", phase1_saved), ("Phase 2 (+ teaching)", phase2_saved)]:
            if res:
                sr = res.get("success_rate", 0)
                print(f"Loaded {label}: {res.get('successes', '?')}/{res.get('num_episodes', '?')} ({sr:.1%})")
        _print_pipeline_summary(task_type, save_name, phase1_saved, phase2_saved)
        return phase1_saved, phase2_saved

    # --- Load abstract elements and lexicons ---
    with open(elements_path) as f:
        rule_spec = json.load(f)

    from herosjourney.core.elements import fill_elements, load_lexicons, save_variant as _save_variant
    sem_lex, nonce_lex = load_lexicons(rule_spec.get("lexicon", "default"))

    # Pre-load distractor spec once (fill_elements is called per-variant inside the loop)
    _dist_spec = None
    _dist_sem_lex = _dist_nonce_lex = None
    if distractor_rules_path:
        with open(distractor_rules_path) as _f:
            _dist_spec = json.load(_f)
        _dist_sem_lex, _dist_nonce_lex = load_lexicons(_dist_spec.get("lexicon", "default"))

    # --- Set up variants directory ---
    if variants_dir is None:
        variants_dir = os.path.join(VARIANTS_DIR, task_type)
    os.makedirs(variants_dir, exist_ok=True)

    # --- Variant loop ---
    p1_variant_list:  List[Dict] = []
    p11_variant_list: List[Dict] = []
    p2_variant_list:  List[Dict] = []

    for variant_seed in range(num_variants):
        if num_variants > 1:
            print(f"\n{'='*60}")
            print(f"VARIANT {variant_seed + 1}/{num_variants}  (seed={variant_seed})")
            print(f"{'='*60}\n")

        # Fill abstract elements with surface names for this seed
        elements = fill_elements(rule_spec, sem_lex, nonce_lex, seed=variant_seed,
                                 split_spec=spec.split)
        variant_path = os.path.join(variants_dir, f"v{variant_seed}_filled.json")
        _save_variant(elements, variant_path)

        # Generate source and gen tasks
        source_tasks = gen_fn(elements, split="source", use_nonce=use_nonce)
        gen_tasks    = gen_fn(elements, split="gen",    use_nonce=use_nonce)
        print(f"Loaded {len(source_tasks)} source tasks and {len(gen_tasks)} gen tasks ({task_type})")

        all_demo_source_tasks = list(source_tasks)

        # Shuffle demo tasks reproducibly
        _random.Random(variant_seed).shuffle(all_demo_source_tasks)

        # Coverage sweep: subset demos and gen tasks
        demo_source_tasks = (all_demo_source_tasks[:n_source_demos]
                             if n_source_demos is not None else all_demo_source_tasks)
        eval_gen_tasks    = (gen_tasks[:n_gen_tasks]
                             if n_gen_tasks is not None else gen_tasks)
        if n_source_demos is not None or n_gen_tasks is not None:
            print(f"  Coverage: {len(demo_source_tasks)} source demos, {len(eval_gen_tasks)} gen tasks")

        # Build demos
        if distractor_rules_path:
            _dist_tmpl    = get_template(spec.distractor_template or spec.template_name)
            _dist_elems   = fill_elements(_dist_spec, _dist_sem_lex, _dist_nonce_lex, seed=variant_seed)
            _all_dist     = generate_tasks(_dist_elems, process=None, split=None,
                                           use_nonce=use_nonce, task_type="distractor",
                                           template_name=_dist_tmpl.name, seed=variant_seed)
            distractor_tasks = (
                _all_dist if num_distractor_samples >= len(_all_dist)
                else _random.Random(variant_seed).sample(_all_dist, num_distractor_samples)
            )
            demos = generate_mixed_demos(
                property_tasks=demo_source_tasks,
                distractor_tasks=distractor_tasks,
                gen_tasks=gen_tasks,
                initial_currency=initial_currency,
                seed=variant_seed,
            )
            print(
                f"Built {len(demos)} demos "
                f"({len(demo_source_tasks)} property + {len(distractor_tasks)} distractor)"
            )
        else:
            demos = generate_demos(
                demo_source_tasks,
                initial_currency=initial_currency,
                seed=variant_seed,
            )
            world_listing = build_world_listing(source_tasks + gen_tasks)
            for d in demos:
                d.metadata["world_listing"] = world_listing
            print(f"Built {len(demos)} property demos (no distractors, {demo_repeats} repeat(s))")

        demo_context = construct_demo_context(demos, include_world=True)

        # --- Phase 1: demo only ---
        if run_p1:
            print(f"\nPhase 1: running {len(eval_gen_tasks)} episodes...")
            raw1 = run_episodes_batch(
                gen_tasks=eval_gen_tasks,
                demo_context=demo_context,
                max_runs=max_runs,
                verbose=verbose,
                truncate_window=truncate_window,
                model_path=model_path,
                initial_currency=initial_currency,
                num_workers=num_workers,
                converter_model=converter_model,
                teaching_message="",
                phase_label=f"Phase 1 v{variant_seed}",
                num_tries=num_tries,
                variant_seed=variant_seed,
                source_tasks=demo_source_tasks,
                no_reasoning=no_reasoning,
            )
            eps1, succ1, tok1 = _format_batch_results(raw1, "ep", model_path)
            v1 = _phase_summary(eps1, succ1, tok1, variant_seed=variant_seed)
            p1_variant_list.append(v1)
            print(f"  Variant {variant_seed}: {succ1}/{len(eps1)} succeeded ({v1['success_rate']:.1%})")
            _print_token_usage(tok1)

        # --- Phase 1.1: curate teaching message for Phase 2 ---
        variant_teaching_message: Union[str, List[str]] = ""
        if run_p2 and teaching_strategy != "none":
            from herosjourney.runner.teacher import (
                STATIC_STRATEGIES, DYNAMIC_STRATEGIES, Teacher,
                build_static_teaching_messages,
            )

            # Check if we already have a saved Phase 1.1 result for this variant
            saved_variant_11 = None
            if phase1_1_saved:
                v11_list = phase1_1_saved.get("variants", [phase1_1_saved])
                if variant_seed < len(v11_list):
                    saved_variant_11 = v11_list[variant_seed]
                elif v11_list:
                    saved_variant_11 = v11_list[0]

            if saved_variant_11 is not None:
                variant_teaching_message = saved_variant_11.get("teaching_message", "")
                print(f"  Loaded Phase 1.1 teaching message from file ({len(variant_teaching_message)} chars)")
            elif teaching_strategy in STATIC_STRATEGIES:
                tree_list = [(task.tree, task.tree.root_id) for task in gen_tasks]
                variant_teaching_message = build_static_teaching_messages(
                    teaching_strategy, tree_list
                )
                print(f"  Built {len(variant_teaching_message)} static teaching messages ({teaching_strategy})")
            elif teaching_strategy in DYNAMIC_STRATEGIES:
                # Use the current variant's Phase 1 summary if just run, else load from saved file.
                if run_p1:
                    p1_for_curation = v1
                else:
                    variants_list = (phase1_saved or {}).get("variants", [])
                    p1_for_curation = (
                        variants_list[variant_seed]
                        if variant_seed < len(variants_list)
                        else variants_list[0] if variants_list
                        else (phase1_saved or {})
                    )

                teacher = Teacher.from_strategy(
                    teaching_strategy,
                    teacher_model_path or model_path,  # default: same model as student
                )
                print(f"  Curating teaching message via '{teaching_strategy}'...")
                curated_msg, curation_history = teacher.curate(
                    phase1_gen_results=p1_for_curation,
                    verbose=verbose,
                    converter_model=converter_model,
                )
                variant_teaching_message = curated_msg
                print(f"  Curation complete. Teaching message length: {len(curated_msg)} chars")
                p11_variant_list.append({
                    "variant_seed":    variant_seed,
                    "strategy":        teaching_strategy,
                    "teaching_message": curated_msg,
                    "curation_history": curation_history,
                })

        # --- Phase 2: demo + teaching ---
        if run_p2:
            # hr and idea use modified episode runners; others use standard.
            p2_episode_mode = (
                teaching_strategy
                if teaching_strategy in {"hr", "idea"}
                else "standard"
            )
            print(f"\nPhase 2: running {len(eval_gen_tasks)} episodes (mode={p2_episode_mode})...")
            raw2 = run_episodes_batch(
                gen_tasks=eval_gen_tasks,
                demo_context=demo_context,
                max_runs=max_runs,
                verbose=verbose,
                truncate_window=truncate_window,
                model_path=model_path,
                initial_currency=initial_currency,
                num_workers=num_workers,
                converter_model=converter_model,
                teaching_message=variant_teaching_message,
                phase_label=f"Phase 2 v{variant_seed}",
                num_tries=num_tries,
                variant_seed=variant_seed,
                source_tasks=demo_source_tasks,
                episode_mode=p2_episode_mode,
                no_reasoning=no_reasoning,
            )
            eps2, succ2, tok2 = _format_batch_results(raw2, "ep", model_path)
            v2 = _phase_summary(eps2, succ2, tok2, variant_seed=variant_seed)
            p2_variant_list.append(v2)
            print(f"  Variant {variant_seed}: {succ2}/{len(eps2)} succeeded ({v2['success_rate']:.1%})")
            _print_token_usage(tok2)

    # --- Aggregate and save ---
    if run_p1:
        phase1_saved = _aggregate_variant_results(p1_variant_list, num_variants, demo_repeats)
        save_phase_results(phase1_saved, "1", save_name)

    if p11_variant_list:
        phase1_1_agg = {
            "strategy":   teaching_strategy,
            "num_variants": num_variants,
            "variants":   p11_variant_list,
            # Top-level teaching_message = first variant (single-variant common case)
            "teaching_message": p11_variant_list[0]["teaching_message"] if p11_variant_list else "",
        }
        save_phase1_1_results(phase1_1_agg, save_name)

    if run_p2:
        phase2_saved = _aggregate_variant_results(p2_variant_list, num_variants, demo_repeats)
        save_phase_results(phase2_saved, "2", save_name)

    _print_pipeline_summary(task_type, save_name, phase1_saved, phase2_saved)
    return phase1_saved, phase2_saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_token_usage(tok: Dict) -> None:
    if tok.get("total_tokens", 0) > 0:
        print(
            f"  Tokens: {tok['total_prompt_tokens']:,} in + "
            f"{tok['total_completion_tokens']:,} out = {tok['total_tokens']:,} total"
        )
        if tok["estimated_cost_usd"] > 0:
            print(f"  Estimated cost: ${tok['estimated_cost_usd']:.4f}")


def _print_pipeline_summary(
    task_type: str,
    save_name: str,
    phase1: Optional[Dict],
    phase2: Optional[Dict],
) -> None:
    print("\n" + "=" * 60)
    print(f"PIPELINE SUMMARY  [{task_type}  |  {save_name}]")
    print("=" * 60)
    for label, res in [("Phase 1 (demo only)", phase1), ("Phase 2 (+ teaching)", phase2)]:
        if res:
            nv  = res.get("num_variants", 1)
            sr  = res.get("success_rate", 0)
            msg = f"  {label}: {res['successes']}/{res['num_episodes']}  ({sr:.1%})"
            if nv > 1:
                mean = res.get("mean_variant_success_rate", sr)
                std  = res.get("std_variant_success_rate", 0)
                msg += f"  [mean/variant: {mean:.1%} ± {std:.1%}]"
            print(msg)
        else:
            print(f"  {label}: skipped / not run")
    print()


# ---------------------------------------------------------------------------
# Q&A pipeline
# ---------------------------------------------------------------------------

def run_qa_pipeline(
    task_type: str,
    elements_path: str,
    model_path: str,
    qa_modes: List[str],          # subset of ["instance", "structure_mc", "structure_exp"]
    distractor_rules_path: Optional[str] = None,
    num_distractor_samples: int = 4,
    use_nonce: bool = False,
    num_variants: int = 1,
    demo_repeats: int = 1,
    variants_dir: Optional[str] = None,
    n_source_demos: Optional[int] = None,
    n_gen_tasks: Optional[int] = None,
    converter_model: str = "small",
    num_workers: int = 1,
    verbose: bool = False,
    save_name: Optional[str] = None,
    # Two-phase teaching
    teaching_strategy: str = "none",
    teacher_model_path: Optional[str] = None,
    phase1_1_results_file: Optional[str] = None,
    fixed_teaching_message: Optional[str] = None,
    skip_qa_phase1: bool = False,
    skip_qa_phase2: bool = False,
    judge_model_path: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Two-phase Q&A pipeline (instance / structure_exp modes); mirrors run_two_phase_pipeline setup."""
    from herosjourney.core.registry import get_task
    from herosjourney.core.generator import get_template, generate_tasks
    from herosjourney.runner.qa_episode import (
        run_qa_instance, run_qa_structure_exp,
    )
    from herosjourney.runner.prompts import QA_BASE_PROMPT

    spec   = get_task(task_type)
    gen_fn = spec.gen_fn

    # Fall back to registry distractor config if not passed explicitly
    if distractor_rules_path is None:
        distractor_rules_path = spec.distractor_rules

    if save_name is None:
        model_stem = os.path.basename(model_path.rstrip("/")).replace(" ", "_")[:20]
        save_name  = f"{task_type}_{model_stem}"

    run_p1 = not skip_qa_phase1
    run_p2 = not skip_qa_phase2 and (
        teaching_strategy != "none" or fixed_teaching_message is not None
    )

    # Load saved Phase 1.1 (episode pipeline teaching message) if provided
    phase1_1_saved = load_phase1_1_results(save_name, phase1_1_results_file)

    with open(elements_path) as f:
        rule_spec = json.load(f)

    from herosjourney.core.elements import fill_elements, load_lexicons
    sem_lex, nonce_lex = load_lexicons(rule_spec.get("lexicon", "default"))

    # Pre-load distractor spec once
    _dist_spec = None
    _dist_sem_lex = _dist_nonce_lex = None
    if distractor_rules_path:
        with open(distractor_rules_path) as _f:
            _dist_spec = json.load(_f)
        _dist_sem_lex, _dist_nonce_lex = load_lexicons(_dist_spec.get("lexicon", "default"))

    if variants_dir is None:
        variants_dir = os.path.join(VARIANTS_DIR, task_type)

    # -------------------------------------------------------------------------
    # Inner helper: run all selected Q&A modes for one variant + teaching msg
    # -------------------------------------------------------------------------
    def _run_qa_variant(
        variant_seed: int,
        demo_context: str,
        source_tasks: List,
        gen_tasks: List,
        teaching_message: str = "",
        _verbose: bool = False,
    ) -> Dict:
        v: Dict[str, Any] = {"variant_seed": variant_seed}

        # Instance Q&A
        if "instance" in qa_modes:
            print(f"  [QA:instance] {len(gen_tasks)} gen tasks")
            inst_results: List[Dict] = []
            if num_workers > 1:
                with ThreadPoolExecutor(max_workers=num_workers) as ex:
                    futures = {
                        ex.submit(
                            run_qa_instance,
                            task=task,
                            demo_context=demo_context,
                            model=model_path,
                            base_prompt=QA_BASE_PROMPT,
                            converter_model=converter_model,
                            teaching_message=teaching_message,
                            verbose=_verbose,
                        ): i
                        for i, task in enumerate(gen_tasks)
                    }
                    for fut in as_completed(futures):
                        try:
                            inst_results.append(fut.result())
                        except Exception as e:
                            print(f"    Instance QA error: {e}")
            else:
                for task in tqdm(gen_tasks, desc="QA:instance"):
                    inst_results.append(run_qa_instance(
                        task=task,
                        demo_context=demo_context,
                        model=model_path,
                        base_prompt=QA_BASE_PROMPT,
                        converter_model=converter_model,
                        teaching_message=teaching_message,
                        verbose=_verbose,
                    ))
            n_c = sum(r["correct"] for r in inst_results)
            acc = n_c / len(inst_results) if inst_results else 0.0
            print(f"  Instance accuracy: {n_c}/{len(inst_results)} ({acc:.1%})")
            v["instance"] = {"results": inst_results, "accuracy": acc,
                             "correct": n_c, "total": len(inst_results)}

        # Structure MC Q&A (disabled)
        # if "structure_mc" in qa_modes:
        #     print("  [QA:structure_mc]")
        #     mc_result = run_qa_structure_mc(
        #         source_tasks=source_tasks, gen_tasks=gen_tasks,
        #         demo_context=demo_context, task_type=task_type,
        #         model=model_path, base_prompt=QA_BASE_PROMPT,
        #         converter_model=converter_model,
        #         teaching_message=teaching_message,
        #         verbose=_verbose,
        #     )
        #     print(f"  Structure MC: predicted={mc_result['predicted_choice']}  "
        #           f"correct={mc_result['correct_choice']}  "
        #           f"{'✓' if mc_result['correct'] else '✗'}")
        #     v["structure_mc"] = mc_result

        # Structure explanation Q&A
        if "structure_exp" in qa_modes:
            print("  [QA:structure_exp]")
            exp_result = run_qa_structure_exp(
                source_tasks=source_tasks, gen_tasks=gen_tasks,
                demo_context=demo_context, task_type=task_type,
                model=model_path, base_prompt=QA_BASE_PROMPT,
                converter_model=converter_model,
                teaching_message=teaching_message,
                verbose=_verbose,
                judge_model=judge_model_path or model_path,
            )
            print(f"  Structure exp overall: {exp_result['overall']:.3f}  "
                  f"judge: {exp_result['judge_reasoning'][:80]}")
            v["structure_exp"] = exp_result

        return v

    # -------------------------------------------------------------------------
    # Per-variant loop
    # -------------------------------------------------------------------------
    p1_variant_results: List[Dict] = []
    p2_variant_results: List[Dict] = []

    for variant_seed in range(num_variants):
        print(f"\n[QA] Variant {variant_seed + 1}/{num_variants}")

        # Build elements for this variant
        elements = fill_elements(rule_spec, sem_lex, nonce_lex, seed=variant_seed,
                                 split_spec=spec.split)

        source_tasks = gen_fn(elements, split="source", use_nonce=use_nonce)
        gen_tasks    = gen_fn(elements, split="gen",    use_nonce=use_nonce)

        # Coverage sweep: apply the same shuffle as the episode pipeline, then subset
        _shuffled_source = list(source_tasks)
        _random.Random(variant_seed).shuffle(_shuffled_source)
        demo_source_tasks = (_shuffled_source[:n_source_demos]
                             if n_source_demos is not None else _shuffled_source)
        eval_gen_tasks    = (gen_tasks[:n_gen_tasks]
                             if n_gen_tasks is not None else list(gen_tasks))

        # Build demo context (identical to episode pipeline)
        if distractor_rules_path:
            _dist_tmpl  = get_template(spec.distractor_template or spec.template_name)
            _dist_elems = fill_elements(_dist_spec, _dist_sem_lex, _dist_nonce_lex, seed=variant_seed)
            _all_dist   = generate_tasks(_dist_elems, process=None, split=None,
                                         use_nonce=use_nonce, task_type="distractor",
                                         template_name=_dist_tmpl.name, seed=variant_seed)
            dist_tasks  = (
                _all_dist if num_distractor_samples >= len(_all_dist)
                else _random.Random(variant_seed).sample(_all_dist, num_distractor_samples)
            )
            demos = generate_mixed_demos(
                demo_source_tasks, dist_tasks,
                gen_tasks=gen_tasks,
                initial_currency=500, seed=variant_seed,
            )
        else:
            demos = generate_demos(demo_source_tasks, initial_currency=500, seed=variant_seed)
            world_listing = build_world_listing(source_tasks + gen_tasks)
            for d in demos:
                d.metadata["world_listing"] = world_listing

        demo_context = construct_demo_context(demos, include_world=True)

        # Phase 1: no teaching
        if run_p1:
            print("  [Phase 1: no teaching]")
            p1_variant_results.append(
                _run_qa_variant(variant_seed, demo_context, demo_source_tasks, eval_gen_tasks, "",
                                _verbose=verbose)
            )

        # Determine teaching message for Phase 2
        if run_p2:
            variant_teaching_message = ""

            # fixed_teaching_message takes priority over all other sources
            if fixed_teaching_message is not None:
                variant_teaching_message = fixed_teaching_message

            # Load from saved Phase 1.1 file (episode pipeline)
            elif phase1_1_saved:
                v11_list = phase1_1_saved.get("variants", [phase1_1_saved])
                v11      = (v11_list[variant_seed] if variant_seed < len(v11_list)
                            else v11_list[0] if v11_list else None)
                if v11:
                    variant_teaching_message = v11.get("teaching_message", "")
                    print(f"  Loaded teaching message from Phase 1.1 ({len(variant_teaching_message)} chars)")

            if fixed_teaching_message is None and not variant_teaching_message:
                from herosjourney.runner.teacher import (
                    STATIC_STRATEGIES, DYNAMIC_STRATEGIES, Teacher,
                    build_static_teaching_messages,
                )
                if teaching_strategy in STATIC_STRATEGIES:
                    tree_list = [(t.tree, t.tree.root_id) for t in gen_tasks]
                    msgs = build_static_teaching_messages(teaching_strategy, tree_list)
                    # build_static_teaching_messages may return a list; use first/shared msg
                    variant_teaching_message = msgs[0] if isinstance(msgs, list) else msgs
                    print(f"  Built static teaching message ({teaching_strategy})")
                elif teaching_strategy in DYNAMIC_STRATEGIES:
                    # Use QA phase 1 results from the current variant to curate a teaching
                    # message (QA-native phase 1.1).  Falls back to empty if QA was not run.
                    p1_qa = (p1_variant_results[variant_seed]
                             if variant_seed < len(p1_variant_results) else None)
                    if p1_qa:
                        from herosjourney.runner.qa_episode import curate_from_qa_results
                        print(f"  Curating QA-based teaching message via '{teaching_strategy}'...")
                        variant_teaching_message = curate_from_qa_results(
                            qa_variant_result=p1_qa,
                            task_type=task_type,
                            model=model_path,
                            converter_model=converter_model,
                            verbose=verbose,
                        )
                        print(f"  Curation complete ({len(variant_teaching_message)} chars)")
                    else:
                        print(
                            f"  WARNING: dynamic strategy '{teaching_strategy}' requested "
                            "but no QA phase 1 results available (run_p1 was False and no "
                            "phase1_1_results_file provided). Skipping Phase 2 curation."
                        )
                        run_p2 = False

            if run_p2:
                print("  [Phase 2: + teaching]")
                p2_variant_results.append(
                    _run_qa_variant(
                        variant_seed, demo_context, demo_source_tasks, eval_gen_tasks,
                        variant_teaching_message,
                        _verbose=verbose,
                    )
                )

    # -------------------------------------------------------------------------
    # Aggregate and save
    # -------------------------------------------------------------------------
    def _aggregate_qa(variant_results: List[Dict]) -> Dict:
        agg: Dict[str, Any] = {
            "task_type":        task_type,
            "use_nonce":        use_nonce,
            "num_variants":     num_variants,
            "qa_modes":         qa_modes,
            "teaching_strategy": teaching_strategy,
            "variants":         variant_results,
        }
        if "instance" in qa_modes:
            all_inst = [r for v in variant_results
                        for r in v.get("instance", {}).get("results", [])]
            n_c = sum(r["correct"] for r in all_inst)
            agg["instance_accuracy"] = n_c / len(all_inst) if all_inst else 0.0

        # if "structure_mc" in qa_modes:
        #     correct_list = [v["structure_mc"]["correct"]
        #                     for v in variant_results if "structure_mc" in v]
        #     agg["structure_mc_accuracy"] = (
        #         sum(correct_list) / len(correct_list) if correct_list else 0.0
        #     )

        if "structure_exp" in qa_modes:
            exp_list = [v["structure_exp"] for v in variant_results if "structure_exp" in v]
            if exp_list:
                agg["structure_exp_overall"]              = sum(r["overall"]                        for r in exp_list) / len(exp_list)
                agg["structure_exp_input_score"]          = sum(r.get("input_score",          0)  for r in exp_list) / len(exp_list)
                agg["structure_exp_output_score"]         = sum(r.get("output_score",         0)  for r in exp_list) / len(exp_list)
                agg["structure_exp_rule_score"]           = sum(r.get("rule_score",           0)  for r in exp_list) / len(exp_list)
                agg["structure_exp_generalization_score"] = sum(r.get("generalization_score", 0)  for r in exp_list) / len(exp_list)
            else:
                agg["structure_exp_overall"] = 0.0
        return agg

    def _save_qa(agg: Dict, phase_label: str) -> str:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = f"{RESULTS_DIR}/qa_{phase_label}_{save_name}.json"
        with open(path, "w") as f:
            json.dump(agg, f, indent=4)
        print(f"Q&A {phase_label} results saved to: {path}")
        return path

    def _print_qa_summary(agg: Dict, label: str) -> None:
        print(f"\n  {label}:")
        if "instance_accuracy"    in agg: print(f"    Instance accuracy:          {agg['instance_accuracy']:.1%}")
        # if "structure_mc_accuracy" in agg: print(f"    Structure MC accuracy:      {agg['structure_mc_accuracy']:.1%}")
        if "structure_exp_overall" in agg:
            print(f"    Structure exp overall (0–1):  {agg['structure_exp_overall']:.3f}")
            print(f"      input_score          (avg 0–2): {agg.get('structure_exp_input_score',  0):.2f}")
            print(f"      output_score         (avg 0–2): {agg.get('structure_exp_output_score', 0):.2f}")
            print(f"      rule_score           (avg 0–2): {agg.get('structure_exp_rule_score',   0):.2f}")
            print(f"      generalization_score (avg 0–2): {agg.get('structure_exp_generalization_score', 0):.2f}")

    qa_phase1 = qa_phase2 = None

    if run_p1 and p1_variant_results:
        qa_phase1 = _aggregate_qa(p1_variant_results)
        _save_qa(qa_phase1, "phase1")

    if run_p2 and p2_variant_results:
        qa_phase2 = _aggregate_qa(p2_variant_results)
        _save_qa(qa_phase2, "phase2")

    print("\n" + "=" * 60)
    print(f"Q&A SUMMARY  [{task_type}  |  {save_name}]")
    print("=" * 60)
    if qa_phase1:
        _print_qa_summary(qa_phase1, "Phase 1 (no teaching)")
    if qa_phase2:
        _print_qa_summary(qa_phase2, f"Phase 2 (+ {teaching_strategy})")
    print()

    return qa_phase1, qa_phase2
