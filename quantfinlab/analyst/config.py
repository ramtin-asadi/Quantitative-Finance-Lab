from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalystConfig:
    root: Path
    context_tokens: int = 24576
    evidence_tokens: int = 7000
    generation_tokens: int = 2300
    prompt_version: str = "analyst-v5"
    base_model: str = "Qwen/Qwen3.5-2B"
    base_revision: str = "15852e8c16360a2fea060d615a32b45270f8a8fc"
    adapter_repo: str = "ramtinasadi/Quantfinlab-Qwen3.5-2B-Financial-Analysis-LoRA"
    gguf_repo: str = "ramtinasadi/Quantfinlab-Qwen3.5-2B-Financial-Analysis-GGUF"

    def __post_init__(self):
        if self.context_tokens != 24576:
            raise ValueError("Project 24 locks the local inference context at 24,576 tokens.")

    @property
    def workspace(self) -> Path:
        return self.root / "workspace/financial_analyst"

    @classmethod
    def from_repo(cls, path: str | Path = ".") -> "AnalystConfig":
        path = Path(path).resolve()
        for parent in [path, *path.parents]:
            if (parent / "quantfinlab").is_dir() and (parent / "pyproject.toml").is_file():
                return cls(parent)
        raise FileNotFoundError("Run from the Quantitative Finance Lab repository.")
