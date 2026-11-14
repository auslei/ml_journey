"""Make ``import llm_finetune`` work from the digit-prefixed section directory.

The directory name ``07_llm`` is not a valid module name, so the importable package
lives one level down at ``07_llm/llm_finetune``. Tests run from the repo root, so this
inserts the section directory on ``sys.path``. Notebooks run with their own directory as
cwd, so the same import resolves there without this file.
"""

import sys
from pathlib import Path

_section_dir = str(Path(__file__).parent)
if _section_dir not in sys.path:
    sys.path.insert(0, _section_dir)
