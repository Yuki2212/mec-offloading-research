"""Run a comparison with explicit settings and an isolated output directory."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import platform

from mecoptimaloffloading.env_multi_mec import SystemParams
from . import compare


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="MECオフロード手法の比較実験")
    parser.add_argument("--config", type=Path, default=root / "configs/boot_1.0.json")
    parser.add_argument("--output", type=Path, help="新規作成する出力ディレクトリ")
    parser.add_argument("--smoke", action="store_true", help="8ユーザ・2MEC・1シナリオで動作確認")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config["system"].update(num_users=8, num_mecs=2)
        config["ga"].update(pop_size=8, generations=2)
        config["num_scenarios"] = 1
    if config["condition_id"] not in (1, 2, 3):
        parser.error("condition_id は 1, 2, 3 のいずれかを指定してください")
    positive = [config["num_scenarios"], config["system"]["num_users"],
                config["system"]["num_mecs"], config["multicore"]["cores_per_mec"],
                config["ga"]["pop_size"], config["ga"]["generations"]]
    if any(not isinstance(x, int) or x < 1 for x in positive):
        parser.error("シナリオ数・ユーザ数・MEC数・コア数・個体数・世代数は正の整数が必要です")
    if not 0 <= config["ga"]["elitism"] < config["ga"]["pop_size"]:
        parser.error("elitism は 0 以上、pop_size 未満にしてください")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    label = "smoke" if args.smoke else "run"
    output = (args.output or root / "outputs" / f"{label}-condition{config['condition_id']}-{stamp}").resolve()
    # Never write generated results into the preserved reference data.
    reference = (root / "data/reference").resolve()
    if output == reference or reference in output.parents:
        parser.error("data/reference は基準データ専用です。別の出力先を指定してください")
    if output.exists():
        parser.error(f"既存結果の上書きを防ぐため、新しい出力先を指定してください: {output}")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "status": "running", "smoke": args.smoke,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "deap")},
        "system_resolved": asdict(SystemParams(**config["system"])),
        "objective": {name: getattr(compare, name) for name in (
            "FAIRNESS_MIN_DELAY", "FAIRNESS_PENALTY_WEIGHT",
            "ACTIVE_MEC_PENALTY_WEIGHT", "SHUTDOWN_MUTATION_PROB")},
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    try:
        compare.main(config, output)
    except BaseException:
        metadata["status"] = "failed"
        raise
    else:
        metadata["status"] = "completed"
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"\nOutput directory: {output}")


if __name__ == "__main__":
    main()
