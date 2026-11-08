"""Let OBP 0.5.7 import without torch.

obp/__init__.py imports every submodule, and obp/utils.py imports torch,
although SyntheticBanditDataset needs neither. This narrows the package
imports and makes the torch import optional; obp.dataset.synthetic is not
modified. Run once after `pip install --no-deps obp==0.5.7`.
"""
import sysconfig
from importlib.metadata import version
from pathlib import Path

assert version("obp") == "0.5.7", version("obp")
root = Path(sysconfig.get_paths()["purelib"]) / "obp"

(root / "__init__.py").write_text("from obp import dataset, types, utils\nfrom obp.version import __version__\n")
(root / "dataset" / "__init__.py").write_text(
    "from obp.dataset.base import BaseBanditDataset\n"
    "from obp.dataset.synthetic import SyntheticBanditDataset, logistic_reward_function\n")

utils = root / "utils.py"
text = utils.read_text(encoding="utf-8")
if "import torch\n" in text and "torch = None" not in text:
    text = text.replace('"""Useful Tools."""\n', '"""Useful Tools."""\nfrom __future__ import annotations\n', 1)
    text = text.replace("import torch\n", "try:\n    import torch\nexcept ModuleNotFoundError:\n    torch = None\n", 1)
    utils.write_text(text, encoding="utf-8")
