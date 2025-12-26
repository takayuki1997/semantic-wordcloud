# 意味的ワードクラウド (Semantic Word Cloud) アルゴリズム解説

このドキュメントでは、`generate_semantic_wordcloud_v6.py` の動作原理を詳しく解説します。

---

## 目次

1. [概要](#1-概要)
2. [処理フロー](#2-処理フロー)
3. [キーコンセプト](#3-キーコンセプト)
4. [各処理の詳細](#4-各処理の詳細)
5. [パラメータの影響](#5-パラメータの影響)
6. [数学的背景](#6-数学的背景)

---

## 1. 概要

### 従来のワードクラウドとの違い

| 項目 | 従来のワードクラウド | 意味的ワードクラウド |
|------|---------------------|---------------------|
| 配置基準 | ランダム or スパイラル | 意味的類似度 |
| 単語間の関係 | 考慮しない | 類似した単語が近くに配置 |
| 技術 | 単純な衝突回避 | 埋め込みベクトル + Force-directed layout |

### このプログラムが行うこと

```
入力テキスト → 形態素解析 → 頻出単語抽出 → 埋め込みベクトル取得
→ 意味的距離計算 → Force-directed配置 → 画像出力
```

---

## 2. 処理フロー

```
┌─────────────────────────────────────────────────────────────┐
│  1. テキスト読み込み                                          │
│     load_text() → Excel/テキストファイルから文字列取得         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 形態素解析・単語抽出                                      │
│     extract_words() → Janomeで名詞を抽出                     │
│     ・ストップワード除外                                      │
│     ・1文字の単語を除外                                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 頻度カウント・上位N語選択                                  │
│     Counter() → 出現回数をカウント                           │
│     most_common(n) → 上位n語を選択                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 埋め込みベクトル取得                                      │
│     get_embeddings() → OpenAI API (text-embedding-3-small)   │
│     ・キャッシュがあれば再利用                                 │
│     ・新規単語のみAPIで取得                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  5. Force-directed レイアウト                                │
│     force_directed_layout()                                  │
│     ・高次元距離行列を計算                                    │
│     ・PCAで初期配置                                          │
│     ・力学シミュレーションで最適化                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  6. 色計算・描画                                              │
│     compute_semantic_colors() → 埋め込みから色相を決定        │
│     render_wordcloud() → matplotlib で描画・保存             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. キーコンセプト

### 3.1 埋め込みベクトル (Embedding Vector)

**単語を高次元の数値ベクトルに変換したもの**

```
「機械学習」 → [0.012, -0.034, 0.056, ..., 0.023]  (1536次元)
「深層学習」 → [0.015, -0.031, 0.058, ..., 0.019]  (1536次元)
「料理」     → [-0.045, 0.067, -0.012, ..., 0.089] (1536次元)
```

- 意味が似ている単語 → ベクトルも似ている（近い）
- 意味が異なる単語 → ベクトルも異なる（遠い）

OpenAI の `text-embedding-3-small` モデルは1536次元のベクトルを出力します。

### 3.2 コサイン類似度 (Cosine Similarity)

**2つのベクトルの「向き」の類似度を測る指標**

```
              A · B
cos(θ) = ─────────────
           |A| × |B|

値の範囲: -1 〜 1
  1: 完全に同じ方向（非常に類似）
  0: 直交（関連なし）
 -1: 正反対の方向（対義的）
```

実際の単語埋め込みでは、ほとんどの値が 0.3〜0.9 の範囲に収まります。

```python
# コード内での計算
sim_matrix = cosine_similarity(embeddings)  # 類似度行列
dist_matrix = 1 - sim_matrix                 # 距離行列に変換
```

### 3.3 PCA (主成分分析)

**高次元データを低次元に圧縮する手法**

```
1536次元 → 2次元

目的: 初期配置の決定
方法: データの分散が最大になる方向を見つけて射影
```

```python
pca = PCA(n_components=2)
initial_coords = pca.fit_transform(embeddings)
```

PCAは「できるだけ情報を保持しながら次元を減らす」ため、
意味的に近い単語は2次元でも近くに配置されます。

### 3.4 初期配置: PCA vs t-SNE

本プログラムではPCAを初期配置に使用していますが、t-SNEも選択肢の一つです。

#### 比較表

| 項目 | PCA | t-SNE |
|------|-----|-------|
| 計算速度 | **高速** | 遅い（反復計算） |
| 決定性 | **決定的**（毎回同じ結果） | 確率的（毎回異なる） |
| 大域構造 | **保持される** | 失われやすい |
| 局所構造 | 部分的に保持 | **よく保持される** |
| クラスタ分離 | 緩やか | **明確に分離** |

#### PCAを選んだ理由

1. **Force-directedとの相性**
   - PCAは大域的な構造を保持 → Force-directedで微調整
   - t-SNEは既に局所最適化済み → Force-directedと役割が重複

2. **決定性**
   - PCAは同じ入力に対して常に同じ出力
   - ランダムシードの管理が単純

3. **計算効率**
   - PCAは線形変換で高速
   - t-SNEはO(N²)〜O(N log N)の反復計算

#### t-SNEを使う場合のメリット・デメリット

**メリット:**
- クラスタがより明確に分離される
- 局所的な類似関係がよく表現される

**デメリット:**
- 計算時間が長い
- 大域的な距離関係が歪む（遠い点同士の距離が信頼できない）
- Force-directedレイアウトの効果が薄れる（t-SNE自体が配置最適化）
- perplexityパラメータの調整が必要

#### コード例（t-SNEを使う場合）

```python
from sklearn.manifold import TSNE

# PCAの代わりにt-SNEを使用
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# initial_coords = tsne.fit_transform(embeddings)
```

**結論:** 本プログラムでは、Force-directedレイアウトが主役であり、
初期配置は「おおまかな配置」を提供すれば十分なため、高速で決定的なPCAを採用しています。

### 3.5 Force-directed Layout (力指向レイアウト)

**物理シミュレーションによるグラフ配置アルゴリズム**

各単語をノード、意味的関係をエッジとみなし、
バネと反発力でノードを動かして最適な配置を見つけます。

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   単語A ←──バネ──→ 単語B                          │
│     ↑                ↑                          │
│     │                │                          │
│   反発力           反発力                         │
│     │                │                          │
│     ↓                ↓                          │
│   単語C            単語D                          │
│                                                  │
└──────────────────────────────────────────────────┘

バネ: 理想距離に近づけようとする力
反発力: 重なりを避けるための力
```

---

## 4. 各処理の詳細

### 4.1 ストップワードの読み込み

```python
def load_stopwords(custom_filepath: str = "stopwords.txt") -> set:
```

**3つのソースからストップワードを収集:**

1. **stopwords-iso 日本語** (~134語)
   - 「これ」「それ」「ある」「いる」など

2. **stopwords-iso 英語** (~1298語)
   - "the", "and", "is", "of" など

3. **カスタムファイル** (stopwords.txt)
   - ユーザー定義の除外単語

**キャッシュ機構:**
- 初回: URLからダウンロード → `stopwords_cache.json` に保存
- 2回目以降: キャッシュから読み込み（高速）

### 4.2 形態素解析

```python
def extract_words(text: str) -> list[str]:
    tokenizer = Tokenizer()  # Janome
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')
        if pos[0] == '名詞' and pos[1] not in ['非自立', '代名詞', '数']:
            # 名詞かつ、非自立・代名詞・数詞でない
            ...
```

**Janomeの出力例:**
```
「機械学習の研究」
  ↓
機械学習 / 名詞,固有名詞,一般,*
の     / 助詞,連体化,*,*
研究   / 名詞,サ変接続,*,*
```

**フィルタリング条件:**
- 品詞が「名詞」である
- 名詞の中でも「非自立」「代名詞」「数」は除外
- 1文字の単語は除外
- ストップワードに含まれていない

### 4.3 埋め込み取得とキャッシュ

```python
def get_embeddings(words: list[str], api_key: str = None,
                   cache_file: str = "embeddings_cache.json") -> np.ndarray:
```

**キャッシュ構造:**
```json
{
  "機械学習": [0.012, -0.034, ...],
  "深層学習": [0.015, -0.031, ...],
  "ロボット": [0.023, 0.045, ...]
}
```

**処理フロー:**
```
1. キャッシュファイルを読み込み
2. 要求された単語のうち、キャッシュにある/ないを分離
3. キャッシュにない単語だけAPIで取得
4. 新しい単語をキャッシュに追加・保存
5. 要求された順序で埋め込みを返す
```

### 4.4 Force-directed Layout の詳細

```python
def force_directed_layout(words, canvas_width, canvas_height, iterations, verbose):
```

#### Step 1: 異方性スケールの計算

```python
aspect_ratio = canvas_width / canvas_height  # 例: 900/700 = 1.286
scale_x = np.sqrt(aspect_ratio)              # 例: 1.134
scale_y = 1.0 / np.sqrt(aspect_ratio)        # 例: 0.882
```

これにより、正方形ではなく楕円形のレイアウトになります。

#### Step 2: 距離行列の計算

```python
sim_matrix = cosine_similarity(embeddings)  # N×N の類似度行列
dist_matrix = 1 - sim_matrix                 # 距離に変換 (0〜2)

# 理想距離にスケーリング
ideal_distances = (dist_matrix / max_dist) * ideal_scale
```

**例:**
```
            機械学習  深層学習  料理
機械学習      0.0      0.15    0.85
深層学習      0.15     0.0     0.82
料理          0.85     0.82    0.0

→ 機械学習と深層学習は近く、料理は遠く配置される
```

#### Step 3: PCAによる初期配置

```python
pca = PCA(n_components=2)
initial_coords = pca.fit_transform(embeddings)

# 異方性スケールを適用
word.x = center_x + initial_coords[i, 0] * scale * scale_x
word.y = center_y + initial_coords[i, 1] * scale * scale_y
```

#### Step 4: 力の計算と更新

**3種類の力と計算次元（重要）:**

このアルゴリズムの核心は、**高次元の意味的距離を2次元配置で再現する**ことです。
各力がどの次元で計算されるかを理解することが重要です。

| 力 | 理想/目標 | 現在/実際 | 目的 |
|---|---|---|---|
| スプリング力 | **1536次元**（コサイン距離） | 2次元（キャンバス） | 意味的関係の再現 |
| 反発力 | - | 2次元のみ | 視覚的重なり防止 |
| 中心引力 | - | 2次元のみ | キャンバス内収束 |

1. **スプリング力 (attraction)** ← 高次元→2次元マッピングの核心
   ```python
   # 理想距離: 1536次元空間でのコサイン距離から算出
   ideal_dist = ideal_distances[i, j]  # 高次元由来

   # 現在距離: 2次元キャンバス上の距離
   current_dist = np.sqrt(dx**2 + dy**2)  # 2次元

   spring_force = attraction_strength * (current_dist - ideal_dist)
   ```
   - 現在距離 > 理想距離 → 引き寄せる
   - 現在距離 < 理想距離 → 押し離す
   - **意味的に近い単語は2Dでも近くに配置される**

2. **反発力 (repulsion)** ← 純粋に2次元の視覚的問題
   ```python
   # すべて2次元で計算
   min_sep = (words[i].width + words[j].width) / 2 + 5  # 単語幅（2D）
   actual_dist = np.sqrt(dx**2 + dy**2)  # 2D距離

   if actual_dist < min_sep:
       repulsion = repulsion_strength / (actual_dist² + 1)
   ```
   - 単語が重なりそうなとき発生
   - 距離が近いほど強い
   - **意味的距離とは無関係、視覚的な配置のみ考慮**

3. **中心への引力 (center force)** ← 2次元
   ```python
   center_force = 0.002 * dist_to_center
   ```
   - 単語が散らばりすぎないように中心に引き寄せる

**シミュレーションループ:**
```python
for iteration in range(iterations):  # デフォルト500回
    forces = compute_forces()
    temperature = max(0.1, 1.0 - iteration / iterations)  # 冷却

    for word in words:
        word.vx = (word.vx + force) * damping * temperature
        word.vy = (word.vy + force) * damping * temperature
        word.x += word.vx
        word.y += word.vy
```

**温度 (temperature):**
- 初期: 1.0 → 大きく動く
- 最終: 0.1 → 微調整のみ
- 徐々に冷却することで安定した配置に収束

### 4.5 色の計算

```python
def compute_semantic_colors(words: list[Word]) -> list[tuple]:
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)

    for x, y in coords_2d:
        angle = np.arctan2(y, x)           # 角度 (-π 〜 π)
        hue = (angle + np.pi) / (2 * np.pi)  # 色相 (0 〜 1)
```

**色相環による着色:**
```
        赤 (0°)
         ↑
  紫 ←───┼───→ 黄
         ↓
       青緑 (180°)
```

意味的に近い単語は似た角度 → 似た色になります。

---

## 5. パラメータの影響

### コマンドラインオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-n, --num-words` | 80 | 表示する単語数 |
| `--cache-words` | 200 | キャッシュする単語数 |
| `--iterations` | 500 | シミュレーション反復回数 |
| `--seed` | None | ランダムシード |
| `-o, --output` | semantic_wordcloud.png | 出力ファイル名 |

### 内部パラメータ

```python
# force_directed_layout 内
attraction_strength = 0.01   # スプリングの強さ
repulsion_strength = 500     # 反発力の強さ
damping = 0.9                # 減衰係数

# キャンバス
canvas_width = 900
canvas_height = 700

# フォントサイズ
font_size = 12 + 28 * (ratio ** 0.5)  # 12〜40の範囲
```

### パラメータ調整のヒント

| 症状 | 調整 |
|------|------|
| 単語が散らばりすぎ | `attraction_strength` を上げる |
| 単語が密集しすぎ | `repulsion_strength` を上げる |
| 配置が安定しない | `iterations` を増やす |
| 重なりが多い | `repulsion_strength` を上げる or `ideal_scale` を下げる |

---

## 6. 数学的背景

### 6.1 コサイン類似度の幾何学的意味

2つのベクトル A, B のコサイン類似度:

```
cos(θ) = (A · B) / (|A| × |B|)

ここで:
  A · B = Σ(ai × bi)  (内積)
  |A| = √Σ(ai²)       (ノルム)
```

**なぜコサイン類似度を使うのか:**
- ベクトルの「大きさ」ではなく「向き」を比較
- 単語の出現頻度に依存しない
- 値が -1〜1 に正規化される

### 6.2 PCAの数学

**目標:** 分散を最大化する方向を見つける

```
1. データの平均を引く（中心化）
2. 共分散行列を計算: C = (1/n) × X^T × X
3. 固有値分解: C × v = λ × v
4. 最大固有値に対応する固有ベクトルが第1主成分
```

1536次元 → 2次元の場合、上位2つの固有ベクトルを使用。

### 6.3 Force-directed Layout の物理モデル

**フックの法則 (バネ):**
```
F = -k × (x - x₀)

k: バネ定数 (attraction_strength)
x: 現在の距離
x₀: 理想距離 (ideal_distance)
```

**クーロンの法則 (反発):**
```
F = k × q₁ × q₂ / r²

簡略化: F = repulsion_strength / (r² + 1)
```

**運動方程式:**
```
v(t+1) = (v(t) + F) × damping × temperature
x(t+1) = x(t) + v(t+1)
```

### 6.4 異方性スケールの意味

```
aspect_ratio = width / height = 900 / 700 ≈ 1.286

scale_x = √(aspect_ratio) ≈ 1.134
scale_y = 1 / √(aspect_ratio) ≈ 0.882

scale_x × scale_y = 1 (面積保存)
```

これにより:
- x方向に1.134倍伸ばす
- y方向に0.882倍縮める
- 全体の「面積」は変わらない
- 結果として横長の楕円形になる

---

## 付録: コード構造

```
generate_semantic_wordcloud_v6.py
│
├── 定数・設定
│   ├── FONT_PATHS
│   ├── JAPANESE_STOPWORDS_URL
│   ├── ENGLISH_STOPWORDS_URL
│   └── STOPWORDS_CACHE_FILE
│
├── クラス
│   └── Word (単語の状態を保持)
│       ├── text, freq, font_size, embedding
│       ├── x, y, vx, vy, rotation
│       ├── width, height (プロパティ)
│       ├── get_bbox()
│       └── overlaps()
│
├── ユーティリティ関数
│   ├── load_stopwords()
│   ├── get_font_path()
│   ├── extract_words()
│   ├── load_text_from_excel()
│   └── load_text()
│
├── コア処理
│   ├── get_embeddings()
│   ├── force_directed_layout()
│   │   ├── compute_forces()
│   │   └── count_overlaps()
│   ├── compute_semantic_colors()
│   └── render_wordcloud()
│
└── main()
    └── 引数解析・処理フロー制御
```

---

## 参考資料

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Force-directed graph drawing (Wikipedia)](https://en.wikipedia.org/wiki/Force-directed_graph_drawing)
- [stopwords-iso](https://github.com/stopwords-iso)
- [Janome](https://mocobeta.github.io/janome/)
