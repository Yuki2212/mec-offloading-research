# 基準データ

保存済みのCSV6個です。通常実験の出力形式と同じです。

| ファイル末尾 | MEC1台あたりの起動エネルギー | 対応設定 |
|---|---:|---|
| `1.csv` | 1.0 J | `configs/boot_1.0.json` |
| `2.csv` | 0.5 J | `configs/boot_0.5.json` |
| `3.csv` | 1.5 J | `configs/boot_1.5.json` |

条件番号と起動エネルギーの対応は設定ファイルで管理しています。

- `results_compare_all_no_capacity*.csv`: 1行＝シナリオ×手法。各400行（100シナリオ×4手法）。
- `results_per_user_long_no_capacity*.csv`: 1行＝シナリオ×手法×ユーザ。各40,000行（上記×100ユーザ）。
- `scenario`・`seed`・`algo` が両CSVの対応キーです。ユーザ別には `user` が加わります。
- `algo`: `GA`、`Greedy`、`CEGA`、`Random`。`CEGA` はコード上のCEGA-like実装の出力ラベルです。
- エネルギーはJ、遅延・締切・実行時間は秒、`viol_rate` は百分率です。
- `dst`: 0〜7がMEC、8がクラウド（保存済みの8MEC条件）。
- `excess = max(delay - deadline, 0)`、`slack = deadline - delay`。
- `no_capacity` は容量違反ペナルティを使わない版の名称です。コアの待ち時間計算は存在します。

新しい実験結果は `outputs/` に保存し、このデータを上書きしないでください。
