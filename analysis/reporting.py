"""CSV loading, plots, and summary tables shared by the four notebooks.

Input frames are never narrowed in place; two-method plots cannot affect later
four-method summaries. Each figure has one output name per report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import NullLocator
import pandas as pd
import seaborn as sns

PLOT_ORDER = ("GA", "Greedy", "CEGA", "Random")
TABLE_ORDER = ("GA", "CEGA", "Greedy", "Random")
DISPLAY_NAMES = {"GA": "FBE-CEGA"}
PALETTE = dict(zip(PLOT_ORDER, ("#A7C7E7", "#B7E4C7", "#FAD2E1", "#FFE5B4")))
METRICS = ("energy", "viol_rate", "max_delay", "avg_delay", "avg_ue_energy",
           "E_mec", "E_cloud", "jain_delay")
TABLE_GROUPS = {
    "energy": ("energy", "avg_ue_energy", "E_mec", "E_cloud"),
    "qoe": ("avg_delay", "violations", "jain_delay", "max_delay"),
    "runtime": ("runtime_sec",),
}
LABELS = {
    "ja": {
        "algorithm": "アルゴリズム", "energy": "総消費エネルギー量 [J]",
        "viol_rate": "締切違反率 [%]", "max_delay": "最大遅延時間 [s]",
        "avg_delay": "平均遅延時間 [s]", "avg_ue_energy": "ユーザ端末平均消費エネルギー量 [J]",
        "E_mec": "MEC消費エネルギー量 [J]", "E_cloud": "クラウド消費エネルギー量 [J]",
        "num_active_mec": "起動MEC数 [台]", "jain_delay": "遅延時間のJain公平度",
        "runtime_sec": "実行時間 [s]", "excess": "締切超過時間 [s]",
        "total_delay": "総遅延時間 [s]", "violations": "締切違反ユーザ数",
    },
    "en": {
        "algorithm": "Algorithm", "energy": "Total energy consumption [J]",
        "viol_rate": "Deadline violation rate [%]", "max_delay": "Maximum delay [s]",
        "avg_delay": "Average delay [s]", "avg_ue_energy": "Average user-device energy consumption [J]",
        "E_mec": "MEC energy consumption [J]", "E_cloud": "Cloud energy consumption [J]",
        "num_active_mec": "Number of active MEC servers", "jain_delay": "Jain fairness index of delay",
        "runtime_sec": "Runtime [s]", "excess": "Deadline excess [s]",
        "total_delay": "Total delay [s]", "violations": "Number of deadline-violating users",
    },
}


@dataclass
class ExperimentData:
    runs: pd.DataFrame
    users: pd.DataFrame

    @classmethod
    def load(cls, directory: Path, condition: int) -> ExperimentData:
        """Read each CSV once; retain the four experiment methods."""
        if condition not in (1, 2, 3):
            raise ValueError("condition must be 1, 2, or 3")
        directory = Path(directory)
        runs = pd.read_csv(directory / f"results_compare_all_no_capacity{condition}.csv")
        users = pd.read_csv(directory / f"results_per_user_long_no_capacity{condition}.csv")
        required_runs = set(METRICS) | {"algo", "num_active_mec", "runtime_sec", "violations"}
        required_users = {"algo", "delay", "deadline"}
        for name, frame, required in (("run", runs, required_runs), ("user", users, required_users)):
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")
        runs = runs.loc[runs["algo"].isin(PLOT_ORDER)].copy()
        users = users.loc[users["algo"].isin(PLOT_ORDER)].copy()
        if "excess" not in users:
            users["excess"] = (users["delay"] - users["deadline"]).clip(lower=0)
        return cls(runs, users)

    def summary(self, group: str, stat: str = "mean") -> pd.DataFrame:
        """Keep the original table order and population variance (ddof=0)."""
        grouped = self.runs.groupby("algo")[list(TABLE_GROUPS[group])]
        if stat == "mean":
            result = grouped.mean()
        elif stat == "var":
            result = grouped.var(ddof=0)
        else:
            raise ValueError("stat must be mean or var")
        return result.reindex(TABLE_ORDER)


class AnalysisReport:
    def __init__(self, data: ExperimentData, output_dir: Path, language: str = "ja",
                 formats: tuple[str, ...] = ("pdf",), show: bool = True):
        if language not in LABELS:
            raise ValueError("language must be ja or en")
        if not formats or len(set(formats)) != len(formats) or set(formats) - {"pdf", "png"}:
            raise ValueError("formats must contain unique pdf/png entries")
        self.data = data
        self.output_dir = Path(output_dir)
        self.language = language
        self.labels = LABELS[language]
        self.formats = formats
        self.show = show
        self.generated: list[Path] = []
        fonts = ["DejaVu Sans", "Arial", "Helvetica"]
        if self.language == "ja":
            fonts = ["IPAexGothic", "IPAPGothic", "Noto Sans CJK JP", "Noto Sans JP",
                     "Yu Gothic", "Meiryo", "Hiragino Sans", "TakaoGothic", *fonts]
        self.fonts = fonts

    def _figure(self):
        # rc_context avoids changing styling in other notebooks or reports.
        with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif": self.fonts,
                             "axes.unicode_minus": False}):
            return plt.subplots(figsize=(8, 6), dpi=150)

    def _finish(self, fig, ax, name, metric, order=PLOT_ORDER, log=False):
        ax.set_xlabel(self.labels["algorithm"], fontsize=20, labelpad=8)
        ax.set_ylabel(self.labels[metric], fontsize=20, labelpad=8)
        ax.set_xticks(range(len(order)), [DISPLAY_NAMES.get(a, a) for a in order])
        if log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
        ax.set_title("")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.2)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="both", which="major", direction="in", labelsize=18)
        for text in fig.findobj(Text):
            text.set_fontfamily(self.fonts)
        fig.tight_layout()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
                for extension in self.formats:
                    path = self.output_dir / f"{name}.{extension}"
                    fig.savefig(path, bbox_inches="tight", dpi=300)
                    if path not in self.generated:
                        self.generated.append(path)
            if self.show:
                plt.show()
        finally:
            plt.close(fig)

    def _distribution(self, frame, metric, name, *, kind="violin", order=PLOT_ORDER,
                      log=False, inner="quartile"):
        selected = frame.loc[frame["algo"].isin(order)].copy()
        if log:
            selected = selected.loc[selected[metric] > 0]
        fig, ax = self._figure()
        if not selected.empty:
            common = dict(data=selected, x="algo", y=metric, order=list(order),
                          hue="algo", hue_order=list(order), palette=PALETTE,
                          legend=False, linewidth=1.2, ax=ax)
            if kind == "box":
                sns.boxplot(**common, width=0.55, showfliers=False)
            else:
                sns.violinplot(**common, inner=inner, cut=0)
        else:
            ax.text(0.5, 0.5, "No matching samples", transform=ax.transAxes, ha="center")
        self._finish(fig, ax, name, metric, order, log)

    def plot_metrics(self):
        """One violin per continuous metric; active MEC count is shown as a box."""
        for metric in METRICS:
            self._distribution(self.data.runs, metric, f"{metric}_violin")
        self._distribution(self.data.runs, "num_active_mec", "num_active_mec_box", kind="box")

    def plot_user_delays(self):
        """One total-delay bar and one excess box (only users with excess > 0)."""
        totals = self.data.users.groupby("algo")["delay"].sum().reindex(PLOT_ORDER)
        fig, ax = self._figure()
        ax.bar(range(len(PLOT_ORDER)), totals, color=list(PALETTE.values()),
               edgecolor="black", alpha=0.75, width=0.65)
        self._finish(fig, ax, "total_delay_sum_bar", "total_delay")
        violators = self.data.users.loc[self.data.users["excess"] > 0]
        self._distribution(violators, "excess", "deadline_excess_box_only_violators", kind="box")

    def plot_runtime(self):
        self._distribution(self.data.runs, "runtime_sec", "runtime_violin_log", log=True, inner="box")

    def plot_alternatives(self, include_runtime=True):
        """Optional views of already-plotted metrics, with distinct filenames."""
        self._distribution(self.data.runs, "num_active_mec", "num_active_mec_violin")
        if include_runtime:
            means = self.data.runs.groupby("algo")["runtime_sec"].mean().reindex(PLOT_ORDER)
            fig, ax = self._figure()
            ax.bar(range(len(PLOT_ORDER)), means, color=list(PALETTE.values()), edgecolor="black")
            self._finish(fig, ax, "runtime_bar_log", "runtime_sec", log=True)
            self._distribution(self.data.runs, "runtime_sec", "runtime_violin_ga_vs_cega_linear",
                               order=("GA", "CEGA"), inner=None)

    def write_tables(self, include_runtime=True) -> dict[str, pd.DataFrame]:
        """Save full-precision CSV plus six-decimal LaTeX, using one aggregator."""
        tables = {}
        specs = [(group, stat) for group in ("energy", "qoe") for stat in ("mean", "var")]
        if include_runtime:
            specs.append(("runtime", "mean"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for group, stat in specs:
            name = f"{group}_{stat}"
            table = self.data.summary(group, stat)
            tables[name] = table
            table.to_csv(self.output_dir / f"{name}.csv")
            header = "手法" if self.language == "ja" else "Method"
            lines = [r"\begin{table}[htbp]", r"\centering",
                     rf"\begin{{tabular}}{{l{'c' * len(table.columns)}}}", r"\hline",
                     " & ".join([header, *[self.labels[c] for c in table.columns]]) + r" \\",
                     r"\hline"]
            for algo, row in table.iterrows():
                lines.append(" & ".join([DISPLAY_NAMES.get(algo, algo),
                                        *[f"{v:.6f}" for v in row]]) + r" \\")
            lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
            (self.output_dir / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return tables
