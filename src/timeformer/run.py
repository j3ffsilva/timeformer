"""
RunManager — gerencia o ciclo de vida de uma execução experimental.

Cada run recebe um ID único (YYYYMMDD_NNN), tem seu próprio diretório
estruturado e contribui com uma linha para runs_index.csv, permitindo
comparar múltiplas execuções.

Estrutura de saída:
  outputs/runs/
    20260523_001/
      config.json          — hiperparâmetros e configuração da run
      B1/
        train_history.json — loss por época
        best.pt            — melhor checkpoint por val_loss
        final.pt           — checkpoint ao fim do treino
      B2a/ ...
      B2b/ ...
      B3/
        train_history.json
        best.pt  final.pt
        memory.pkl         — PrototypeMemory serializada
      results/
        results_table.csv  — métricas por modelo × split
        results_full.json  — dict completo para análise posterior
        ablation_table.json
    runs_index.csv          — uma linha por run, métricas-chave
"""

from __future__ import annotations

import csv
import json
import pickle
from datetime import datetime
from pathlib import Path


RUNS_ROOT    = Path("outputs/runs")
INDEX_FILE   = RUNS_ROOT / "runs_index.csv"
INDEX_FIELDS = [
    "run_id", "date", "epochs", "lr", "batch_size", "d_model",
    "B1_test_acc", "B2b_test_acc", "B3_test_acc",
    "B2b_ambiguous_acc", "B3_ambiguous_acc",
    "B3_continuation_acc", "B3_sign_flip",
    "notes",
]


class RunManager:
    """
    Gerencia um experimento completo: diretórios, configuração e índice de runs.

    Uso típico:
        run = RunManager()
        run.setup(config)
        trainer = MLMTrainer(model, run.model_dir("B1"))
        ...
        run.update_index(results, config)
    """

    def __init__(self, runs_root: str | Path = RUNS_ROOT) -> None:
        self.runs_root = Path(runs_root)
        self.run_id    = self._next_run_id()
        self.run_dir   = self.runs_root / self.run_id
        self._config: dict = {}

    # ── Inicialização ──────────────────────────────────────────────────────

    def setup(self, config: dict) -> "RunManager":
        """Cria diretório da run e salva configuração."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._config = config
        (self.run_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str)
        )
        print(f"Run iniciada: {self.run_id}")
        print(f"Diretório:    {self.run_dir}/")
        return self

    # ── Diretórios ────────────────────────────────────────────────────────

    def model_dir(self, model_name: str) -> Path:
        """Diretório para checkpoints e histórico de um modelo."""
        d = self.run_dir / model_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def results_dir(self) -> Path:
        """Diretório para os arquivos de resultado da avaliação."""
        d = self.run_dir / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Persistência de memória B3 ────────────────────────────────────────

    def save_memory(self, model_name: str, memory) -> Path:
        """Serializa PrototypeMemory em {model_dir}/memory.pkl."""
        path = self.model_dir(model_name) / "memory.pkl"
        with open(path, "wb") as f:
            pickle.dump(memory, f)
        return path

    def load_memory(self, model_name: str, prefer_best: bool = True):
        """
        Carrega PrototypeMemory do model_dir.

        Se prefer_best=True (padrão), tenta memory_best.pkl primeiro (alinhado
        com best.pt). Cai para memory.pkl (final de treino) se não existir.
        """
        model_d = self.model_dir(model_name)
        candidates = (["memory_best.pkl", "memory.pkl"] if prefer_best
                      else ["memory.pkl"])
        for fname in candidates:
            path = model_d / fname
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
        return None

    # ── Índice de runs ────────────────────────────────────────────────────

    def update_index(self, results: dict, notes: str = "") -> None:
        """
        Adiciona uma linha ao runs_index.csv com as métricas-chave desta run.

        results: dict retornado por Evaluator.evaluate_all()
        """
        self.runs_root.mkdir(parents=True, exist_ok=True)

        def _acc(model: str, split: str) -> str:
            try:
                v = results[model][split]["probe_subj"]["accuracy"]
                return f"{v:.4f}"
            except (KeyError, TypeError):
                return ""

        def _sfr(model: str) -> str:
            try:
                v = results[model]["contrastive"]["sign_flip_rate"]
                return f"{v:.4f}"
            except (KeyError, TypeError):
                return ""

        row = {
            "run_id":              self.run_id,
            "date":                datetime.now().strftime("%Y-%m-%d %H:%M"),
            "epochs":              self._config.get("epochs", ""),
            "lr":                  self._config.get("lr", ""),
            "batch_size":          self._config.get("batch_size", ""),
            "d_model":             self._config.get("d_model", ""),
            "B1_test_acc":         _acc("B1",  "test"),
            "B2b_test_acc":        _acc("B2b", "test"),
            "B3_test_acc":         _acc("B3",  "test"),
            "B2b_ambiguous_acc":   _acc("B2b", "ambiguous_test"),
            "B3_ambiguous_acc":    _acc("B3",  "ambiguous_test"),
            "B3_continuation_acc": _acc("B3",  "continuation"),
            "B3_sign_flip":        _sfr("B3"),
            "notes":               notes,
        }

        write_header = not INDEX_FILE.exists()
        with open(INDEX_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"runs_index.csv atualizado: {INDEX_FILE}")

    # ── Carregamento de runs existentes ───────────────────────────────────

    @classmethod
    def load(cls, run_id: str, runs_root: str | Path = RUNS_ROOT) -> "RunManager":
        """Carrega um RunManager para uma run existente pelo ID."""
        manager = cls.__new__(cls)
        manager.runs_root = Path(runs_root)
        manager.run_id    = run_id
        manager.run_dir   = manager.runs_root / run_id
        config_path = manager.run_dir / "config.json"
        manager._config = json.loads(config_path.read_text()) if config_path.exists() else {}
        return manager

    @classmethod
    def load_latest(cls, runs_root: str | Path = RUNS_ROOT) -> "RunManager":
        """Carrega o RunManager da run mais recente."""
        runs_root = Path(runs_root)
        runs = sorted(
            [d for d in runs_root.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.name,
        )
        if not runs:
            raise FileNotFoundError(f"Nenhuma run encontrada em {runs_root}")
        return cls.load(runs[-1].name, runs_root)

    @staticmethod
    def list_runs(runs_root: str | Path = RUNS_ROOT) -> list[dict]:
        """Lista todas as runs com suas métricas-chave do índice."""
        index = Path(runs_root) / "runs_index.csv"
        if not index.exists():
            return []
        with open(index, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def checkpoint_path(self, model_name: str, which: str = "best") -> Path:
        """Retorna o caminho do checkpoint (best ou final) de um modelo."""
        return self.model_dir(model_name) / f"{which}.pt"

    # ── Representação ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"RunManager(run_id={self.run_id!r}, dir={self.run_dir})"

    # ── Interno ───────────────────────────────────────────────────────────

    def _next_run_id(self) -> str:
        today = datetime.now().strftime("%Y%m%d")
        self.runs_root.mkdir(parents=True, exist_ok=True)
        existing = sorted(
            d.name for d in self.runs_root.iterdir()
            if d.is_dir() and d.name.startswith(today)
        )
        if not existing:
            return f"{today}_001"
        last = existing[-1]
        try:
            n = int(last.split("_")[1]) + 1
        except (IndexError, ValueError):
            n = len(existing) + 1
        return f"{today}_{n:03d}"
