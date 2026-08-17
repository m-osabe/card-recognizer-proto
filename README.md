# TCG カード画像認識・種類判別システム (Card Recognizer)

スマートフォンの写真やカメラ画像から、手元のカードを自動検出・幾何補正（正面化）し、登録済みのマスターカード（数百枚規模）の中から高精度にカードの種類を特定するPythonシステムです。

---

## 📚 ドキュメント

- [**アルゴリズム設計・技術選定書 (docs/ALGORITHM_DESIGN.md)**](docs/ALGORITHM_DESIGN.md)
  - なぜ「ゼロからのディープラーニング学習」を避けたのか
  - 階層型ハイブリッド枠検出（Bottom-Up ＆ Top-Down SIFT逆射影）
  - 透視変換、SIFT幾何検証（RANSAC）、大局的特徴量の詳細数式・アルゴリズム解説
  - GPUなし・数百枚規模に対する適合性評価
- [**手持ち・黒縁カード検出改善レポート (docs/HANDHELD_CARD_DETECTION_IMPROVEMENT.md)**](docs/HANDHELD_CARD_DETECTION_IMPROVEMENT.md)
  - 手持ち撮影・黒縁カードで枠検出が失敗していた原因調査と分析経緯
  - 認識駆動型（Top-Down）枠検出への方針転換とホモグラフィ逆射影の実装
  - 検出率 3.0% $\rightarrow$ 89.7%、Top-1正解率 94.9% の改善検証データ

---

## 🌟 特徴

- **GPU不要・完全CPU動作**：OpenCV SIFT特徴点照合 ＋ 局所テクスチャ・カラーベクトル類似度検索を組み合わせ、一般的なPCのCPUで軽快に動作します。
- **事前学習（トレーニング）不要**：新しいカードを追加する際は、各カード**1枚の画像**（公式画像やスキャン画像）を `data/master/` に置くだけで即座に認識対象になります。
- **手持ち撮影・黒縁カード・斜め歪みへの圧倒的耐性**：
  1. **Bottom-Up枠検出**: 明瞭な画像からカードの四隅を高速自動検出。
  2. **Top-Down SIFT逆射影（新機能）**: 指の重なりや黒縁で前処理が失敗した場合でも、元画像全体からSIFT幾何照合を行い、ホモグラフィ行列 $H^{-1}$ から写真上の正確な四隅を幾何学的に逆算・復元。
- **3画面比較レポートの自動生成**：「入力写真（検出枠付き）」「歪み補正された正面カード」「特定されたマスターカード」を1枚にまとめた視覚レポート画像を `output/` に自動保存。

---

## 📁 ディレクトリ構成

```
card-recognizer/
├── card_detector.py      # 写真からカード四隅を検出・正面に射影変換（歪み補正）
├── matcher_sift.py       # OpenCV SIFT による特徴点照合 & ホモグラフィ逆射影
├── matcher_embedding.py  # カラー・テクスチャ多次元ベクトルによる類似度検索
├── card_engine.py        # 階層型検出・照合・アンサンブル・可視化を統合するメインエンジン
├── cli.py                # コマンドライン実行ツール
├── sample_generator.py   # デモ用サンプルカード＆斜め撮影テスト写真の自動生成
├── README.md             # 本説明書
├── docs/                 # 技術設計書・改善レポート
│   ├── ALGORITHM_DESIGN.md
│   └── HANDHELD_CARD_DETECTION_IMPROVEMENT.md
├── scripts/              # 調査・診断・デバッグ用スクリプト
│   ├── diagnose_dataset.py          # データセット一括診断
│   └── visualize_detection_debug.py # 二値化・輪郭デバッグ可視化
├── data/
│   ├── master/           # 【マスター画像】既知のカードの正面画像 (1カード1枚)
│   ├── test/             # 【テスト画像】スマホ等で撮影した写真
│   └── index/            # 特徴量キャッシュ（自動生成）
└── output/               # 照合結果の可視化画像保存先
```

---

## 🛠️ 環境構築・セットアップ

Python 3.8以上がインストールされている環境で実行してください。

### 1. 仮想環境（venv）の作成と有効化

```bash
# 1. 仮想環境の作成
python -m venv .venv
```

**仮想環境の有効化（OS別）:**

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  > **Note**: PowerShellでスクリプト実行権限エラーが出る場合は、先に `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` を実行してください。

- **Windows (コマンドプロンプト cmd.exe):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. 依存ライブラリのインストール

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 クイックスタート

### 1. デモサンプルで即座に動作確認する
手元にカード画像がなくても、自動生成されたカードと撮影写真で即座に実験できます。

```bash
# デモカード生成 ＋ インデックス構築を一度に実行
python cli.py setup-demo

# テスト画像全件に対する一括認識テスト・正解率レポートを表示
python cli.py evaluate
```

### 2. 単一の写真を判定する
```bash
python cli.py identify data/test/card_01_0000.jpg
```
判定結果と確信度スコア、上位候補が表示され、`output/` に比較画像が出力されます。

---

## 🃏 お手持ちの実物カードで実験する方法

1. **マスター画像の配置**:
   - `data/master/` フォルダに、既知のカードの正面画像（公式画像・スキャン画像など）を保存します。
   - ファイル名がカード名やIDとして認識されます（例: `card-1.jpg`, `Charizard.png`）。

2. **インデックスの構築**:
   ```bash
   python cli.py build-index
   ```
   数百枚のカードでも数秒〜数十秒で特徴量が抽出され、キャッシュされます。

3. **スマホで撮影した写真を判定**:
   - 撮影したカード写真を `data/test/` などに置き、判定コマンドを実行します。
   ```bash
   # 1枚判定
   python cli.py identify data/test/my_photo.jpg

   # フォルダ内を一括評価（例: 20枚）
   python cli.py evaluate --test-dir data/test --limit 20
   ```

---

## ⚙️ 照合アルゴリズムの切り替え

`--method` オプションで照合ロジックを変更できます：
- `--method ensemble` (デフォルト): SIFT幾何整合性 ＋ カラー・テクスチャ類似度の統合判定。最も高精度。
- `--method sift`: SIFT特徴点・RANSACインライア数のみによる判定。
- `--method embedding`: 大局的な色彩・テクスチャベクトル類似度のみによる判定。
