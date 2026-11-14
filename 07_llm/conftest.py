"""Make ``import llm_finetune`` work from the digit-prefixed section directory.

The directory name ``07_llm`` is not a valid module name, so the importable package
lives one level down at ``07_llm/llm_finetune``. Tests run from the repo root, so this
inserts the section directory on ``sys.path``. Notebooks run with their own directory as
cwd, so the same import resolves there without this file.
"""

import sys
from importlib.util import find_spec
from pathlib import Path

_section_dir = str(Path(__file__).parent)
if _section_dir not in sys.path:
    sys.path.insert(0, _section_dir)

# Skip notebooks and module tests unless the llm extra (transformers + peft) is present,
# so `uv run pytest` without the extra does not collect and fail on missing imports.
_have_llm = find_spec("transformers") and find_spec("peft")
collect_ignore_glob = [] if _have_llm else ["*.ipynb", "tests/*"]
