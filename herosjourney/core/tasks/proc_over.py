from pathlib import Path
from herosjourney.core.task_file import load_task_file
load_task_file(Path(__file__).with_suffix(".json"))
