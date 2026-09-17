# 遺伝的アルゴリズムによるMECタスクオフロードの比較実験

複数のMEC（Mobile Edge Computing）サーバとクラウドに対して、各ユーザのタスクをどこで処理するかを決定する研究コードです。
消費エネルギーの削減に加え、締切超過・ユーザ間の遅延の公平性・起動するMEC台数を考慮した遺伝的アルゴリズム（GA）を、Greedy・CEGA-like・Randomと比較します。

このフォルダは、卒研用に保存されていたコードと結果をコピーし、実験条件・解析・生成物を分離して整理したものです。元の研究フォルダには変更を加えていません。

## 研究で扱う問題

各ユーザが持つ計算タスクを、いずれかのMECサーバまたはクラウドへ割り当てます。割当によって通信時間、処理待ち時間、消費エネルギーが変化します。
少数のMECへタスクを集めると起動エネルギーを減らせる一方、処理待ちが増える可能性があります。そのため、エネルギーだけでなく締切と遅延の公平性も評価します。

- **環境生成**：タスクの計算量・入力データ量・締切、ユーザの送信電力・通信路などを乱数seedに基づいて生成します。
- **割当の評価**：端末（UE）、MEC、クラウドのエネルギーと、ユーザごとの遅延を計算します。
- **MEC内の処理**：ユーザID順に、空き時刻の早いコアへタスクを割り当てて処理完了時刻を計算します。現行の評価呼び出しでは各タスク1コアです。
- **起動エネルギー**：実際にタスクを割り当てたMEC台数に、1台あたりの起動エネルギーを掛けて加算します。

これはシミュレーション上のモデルです。実機の計測値を扱うコードではありません。

## 比較する手法

| CSV上の名前 | 内容 |
|---|---|
| `GA` | 起動MEC台数を重視する提案GA。MEC1台の割当を他へ移す突然変異も使用 |
| `Greedy` | 提案GAと同じ目的関数を用いて割当を改善する比較手法 |
| `CEGA` | 遅延とUEエネルギーを考慮するCEGA-like GA。原論文の完全再実装と断定するものではない |
| `Random` | ランダムにオフロード先を選ぶ比較手法 |

提案GAとGreedyの目的関数は、実装では次の和です。

```text
総消費エネルギー
+ 200000 × 全ユーザの締切超過時間の合計 / ユーザ数
+ 10 × max(0.85 − 遅延のJain公平度, 0)
+ 500 × 起動MEC台数
```

Jain公平度は `(Σx)² / (n × Σx²)` で計算します。起動エネルギーの実量と、台数削減を促す目的関数上のペナルティは別の項です。
これらの重みと起動台数削減の突然変異確率（0.35）は `experiments/compare.py` の先頭で定義されています。

## フォルダとファイル

```text
mec-offloading-research/
├── README.md
├── LICENSE                       # ライセンス
├── requirements.txt              # 実験・解析用ライブラリ
├── mecoptimaloffloading/
│   ├── env_multi_mec.py          # 環境・システム設定・乱数生成
│   ├── ga_multi_mec_energy.py    # GA操作と基本評価
│   └── multicore_scheduler.py   # コア割当・待ち時間・起動エネルギー
├── experiments/
│   ├── compare.py               # 4手法の比較・指標集計・CSV出力
│   └── run.py                   # 設定読込・実行・出力先の管理
├── configs/
│   ├── boot_0.5.json
│   ├── boot_1.0.json
│   └── boot_1.5.json
├── analysis/
│   ├── README.md                # 出力一覧・任意表示の設定
│   ├── reporting.py             # 読込・描画・表集計の共通処理
│   ├── analyze01.ipynb          # 1.0 J条件
│   ├── analyze02.ipynb          # 0.5 J条件
│   ├── analyze03.ipynb          # 1.5 J条件
│   └── analyze01_English.ipynb  # 1.0 J条件・英語図
├── data/reference/              # 卒研時の基準CSV6個
├── figures/                     # 採用する図の保存先
└── outputs/                     # 新規実験・解析の生成物
```

比較実験の処理は `experiments/compare.py` に、実験条件は3個のJSON設定に分けています。
`__init__.py` はPythonパッケージとして読み込むためのファイルです。

## 実験条件

| 項目 | 標準設定 |
|---|---|
| ユーザ数・タスク数 | 100 |
| MEC数 | 8 |
| コア数 | 各MEC 4コア |
| シナリオ数 | 100、環境生成seedは1〜100 |
| GA個体数・世代数 | 200個体・300世代 |
| 交叉率・突然変異率 | 0.9・0.08 |
| GA設定のseed | 1 |
| タスク計算量 | 1.0、1.5、2.0 × 10⁹ cycles |
| 入力データ量 | 100、150、200 kB |
| 締切 | 0.10〜0.12秒 |
| 起動エネルギー | 0.5、1.0、1.5 J / 台 / シナリオ |

JSONの `system` は `SystemParams` の上書き設定です。通信・計算資源等の既定値は `env_multi_mec.py` にあります。
実行時には既定値を含むシステム設定を `metadata.json` に保存します。

| 設定 | 元スクリプト | 出力CSV末尾 | 対応する解析 |
|---|---|---|---|
| `boot_1.0.json` | `compare5.py` | `1.csv` | `analyze01` |
| `boot_0.5.json` | `compare5-2.py` | `2.csv` | `analyze02` |
| `boot_1.5.json` | `compare5-3.py` | `3.csv` | `analyze03` |

## 環境の準備

Python 3.10以上を使用し、このREADMEがあるフォルダで実行してください。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` は依存ライブラリの範囲指定です。卒研当時のバージョンを復元した固定環境ではありません。
日本語図では日本語フォントが必要です。環境に応じて `analysis/reporting.py` のフォント候補を調整してください。

## 比較実験の実行

最初に、小規模な動作確認を実行できます。

```bash
python -m experiments.run --config configs/boot_1.0.json --smoke
```

`--smoke` は8ユーザ・2MEC・1シナリオ・8個体・2世代へ縮小します。研究結果の評価には使用しません。

通常の3条件の実験は次のとおりです。各コマンドは100シナリオを実行するため、動作確認より時間がかかります。

```bash
python -m experiments.run --config configs/boot_1.0.json
python -m experiments.run --config configs/boot_0.5.json
python -m experiments.run --config configs/boot_1.5.json
```

出力先は `outputs/run-condition<番号>-<日時>/` です。保存する内容は次の4ファイルです。

- `results_compare_all_no_capacity*.csv`：シナリオ×手法の集計指標。
- `results_per_user_long_no_capacity*.csv`：ユーザ別の遅延・締切・割当情報。
- `config.json`：実際に使用した実験設定。
- `metadata.json`：Python・主要ライブラリのバージョン、システム設定、目的関数の重み、実行状態。

`--output outputs/my-run` で新しい出力先を指定できます。既存ディレクトリへの上書き、および `data/reference/` への出力は拒否します。

## 保存済み結果の解析

実験を再実行せず、コピーした基準CSVから図表を作成できます。

```bash
python -m jupyterlab analysis
```

対象のノートブックを開き、上から順に実行してください。設定セルで `data/reference/` を入力先に、`outputs/analysis/organized/<ノートブック名>/` を出力先に指定します。
4冊は各13セルに整理し、CSV読込・描画・集計は `analysis/reporting.py` を共有しています。CSVは一度だけ読み込み、締切超過図の二重出力を解消しました。
起動MEC台数は箱ひげ図、実行時間は対数バイオリン図を標準とし、別表示は `SHOW_ALTERNATIVES = True` で追加できます。
平均・母分散の集計方法は維持し、表は表示に加えてCSVとLaTeXにも保存します。詳しい出力一覧は [分析コードの説明](analysis/README.md) にあります。

新しい実験を解析する場合は、先頭の設定セルの `DATA_DIR` を、対応する通常実験の出力ディレクトリへ変更します。
たとえば `DATA_DIR = PROJECT_ROOT / "outputs/my-run"` とします。条件番号とノートブックの対応を合わせてください。

主な指標は総エネルギー、MEC・クラウド・UEのエネルギー、平均・最大・95パーセンタイル遅延、締切違反率、起動MEC台数、Jain公平度、計算時間です。
CSVの単位・行数・キーは [基準データの説明](data/reference/README.md) に記載しています。

## 再現性と引き継いだ実装について

基準CSVは過去の実験結果のコピーであり、今回すべての通常実験を再実行して一致を確認したものではありません。実行時間は環境によって変わります。
乱数処理も元の実装を維持しており、環境生成seedとGA設定のseedは別です。
Greedyの初期解とRandomの割当は、元の実験呼び出しでseedを渡さない `np.random.default_rng()` を使います。このため、同じ設定でも毎回同一の結果になるわけではありません。

現行の `evaluate_with_multicore` は基本評価の遅延をマルチコアの遅延へ差し替えますが、基本評価から受け取った `violations` のリストは再計算していません。
したがって、CSVの `violations`・`viol_rate` と、ユーザ別の `delay > deadline` の件数が一致するとは限りません。
整理の段階では計算仕様を変更せず、この点を明記しています。研究上の修正を行う場合は、別の変更として基準結果との差を検証してください。

## 整理時の確認

- 3条件の小規模実験を実行し、それぞれ集計4行・ユーザ別32行のCSV出力を確認しました。
- 検証時だけGreedy・Randomの乱数を揃え、3条件・4手法のCSV出力を確認しました。
- 4冊のノートブックのコードセルを順に実行し、基準CSVからの図表生成を確認しました。確認時は非対話描画を使用しています。
- 基準データの行数・キーの一意性、および実験出力の上書き防止を確認しました。

確認環境：Python 3.10.13、NumPy 2.2.6、DEAP 1.4.3、pandas 2.3.3、Matplotlib 3.10.9、seaborn 0.13.2。
100シナリオ・300世代の通常実験と、過去の基準CSVとの数値一致は未検証です。

ライセンスと利用条件は [LICENSE](LICENSE) を確認してください。
