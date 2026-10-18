"""手写 GPT-2 基线，底层复用 PyTorch 和 Hugging Face。"""

from .config import GPTConfig
from .model import GPT

__all__ = ["GPT", "GPTConfig"]
