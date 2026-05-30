"""
Example 02 — Register and run a custom task from a declarative spec file.

A "task" is just a rule file + a process + a split. Here we reuse the built-in
additive rule but register it under a new name with a different split seed, all
from a JSON spec — no Python task code required.

Run:  python examples/02_custom_task.py
"""
import json
import tempfile
from pathlib import Path

import herosjourney
from herosjourney import get_task
from herosjourney.core.elements import fill_elements, load_lexicons

# Path to a built-in rule file we'll reuse (you'd point this at your own rule).
RULE_PATH = Path(get_task("additive").rules)


def main():
    # Write a declarative task spec. `rules` may be absolute or relative to the spec.
    task_spec = {
        "name": "my_additive_variant",
        "rules": str(RULE_PATH),
        "process": "property_flat",
        "split": {"fn": "two_offset", "seed": 7},
        "max_tries": 5,
        "validate_fn": "validate_additive_split",
        "eval": {
            "correct_rule": "Both attributes contribute independently and additively.",
            "description": "custom additive variant (seed 7)",
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(task_spec, f)
        spec_path = f.name

    # Register from the file (YAML also works if pyyaml is installed).
    herosjourney.register_task(spec_path)
    Path(spec_path).unlink()

    spec = get_task("my_additive_variant")
    print("registered:", spec.task_type, "| max_tries:", spec.max_tries)

    # Generate tasks to confirm it works.
    sem_lex, nonce_lex = load_lexicons()
    with open(spec.rules) as f:
        rule = json.load(f)
    elements = fill_elements(rule, sem_lex, nonce_lex, seed=0, split_spec=spec.split)
    gen_tasks = spec.gen_fn(elements, split="gen", use_nonce=False)
    print(f"generated {len(gen_tasks)} gen entities for the custom task")


if __name__ == "__main__":
    main()
