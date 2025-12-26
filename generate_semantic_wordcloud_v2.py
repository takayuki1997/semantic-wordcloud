#!/usr/bin/env python3
"""Force-directed layoutによる意味的ワードクラウド生成スクリプト"""

import argparse
import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from openai import OpenAI
from janome.tokenizer import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

# 日本語フォントのパス（macOS）
FONT_PATHS = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]

STOPWORDS = {
    '研究', '可能', '手法', '科学', '対象', '利用', '実現', '必要', 'たち',
    'こと', 'もの', 'ため', 'よう', 'それ', 'これ', 'ここ', 'そこ',
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been',
    'jpg', 'png', 'pdf', 'http', 'https', 'www',
}


class Word:
    """単語とその配置情報を保持するクラス"""
    def __init__(self, text: str, freq: int, font_size: float, embedding: np.ndarray):
        self.text = text
        self.freq = freq
        self.font_size = font_size
        self.embedding = embedding
        self.x = 0.0
        self.y = 0.0
        self.rotation = 0  # 0 or 90
        self.vx = 0.0  # 速度
        self.vy = 0.0

    @property
    def width(self) -> float:
        """単語の幅を推定（日本語は文字数×フォントサイズ）"""
        char_width = self.font_size * 1.0
        if self.rotation == 0:
            return len(self.text) * char_width
        else:
            return self.font_size * 1.2

    @property
    def height(self) -> float:
        """単語の高さを推定"""
        if self.rotation == 0:
            return self.font_size * 1.2
        else:
            return len(self.text) * self.font_size * 1.0

    def get_bbox(self) -> tuple:
        """バウンディングボックスを取得 (left, bottom, right, top)"""
        half_w = self.width / 2
        half_h = self.height / 2
        return (self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h)

    def overlaps(self, other: 'Word', padding: float = 2.0) -> bool:
        """他の単語と重なっているか判定"""
        b1 = self.get_bbox()
        b2 = other.get_bbox()
        return not (b1[2] + padding < b2[0] or b2[2] + padding < b1[0] or
                    b1[3] + padding < b2[1] or b2[3] + padding < b1[1])

    def overlap_area(self, other: 'Word') -> float:
        """重なり面積を計算"""
        b1 = self.get_bbox()
        b2 = other.get_bbox()
        x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
        y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        return x_overlap * y_overlap


def get_font_path() -> str:
    for fp in FONT_PATHS:
        if Path(fp).exists():
            return fp
    return None


def extract_words(text: str) -> list[str]:
    tokenizer = Tokenizer()
    words = []
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')
        if pos[0] == '名詞' and pos[1] not in ['非自立', '代名詞', '数']:
            word = token.surface
            if len(word) > 1 and word.lower() not in STOPWORDS and word not in STOPWORDS:
                words.append(word)
    return words


def load_text_from_excel(excel_path: str) -> str:
    xlsx = pd.ExcelFile(excel_path)
    all_text = []
    for sheet_name in xlsx.sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet_name)
        for col in df.columns:
            for value in df[col].dropna():
                if isinstance(value, str):
                    all_text.append(value)
    return '\n'.join(all_text)


def load_text(file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() in ['.xlsx', '.xls']:
        return load_text_from_excel(file_path)
    return path.read_text(encoding='utf-8')


def get_embeddings(words: list[str], api_key: str = None) -> np.ndarray:
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    print(f"OpenAI APIで{len(words)}個の単語の埋め込みを取得中...")

    batch_size = 100
    all_embeddings = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  {min(i+batch_size, len(words))}/{len(words)} 完了")

    return np.array(all_embeddings)


def semantic_compact_layout(
    words: list[Word],
    similarity_matrix: np.ndarray,
    canvas_width: float = 800,
    canvas_height: float = 600,
    iterations: int = 200,
    verbose: bool = True
):
    """意味的配置を保ちながら凝縮するレイアウト"""
    from sklearn.manifold import TSNE

    center_x = canvas_width / 2
    center_y = canvas_height / 2

    # Step 1: t-SNEで意味的に配置
    if verbose:
        print("  Step 1: t-SNEで意味的配置...")
    embeddings = np.array([w.embedding for w in words])
    perplexity = min(30, len(words) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    # 座標を正規化（中心を0,0に）
    coords[:, 0] -= coords[:, 0].mean()
    coords[:, 1] -= coords[:, 1].mean()

    # 初期スケール（大きめに配置）
    scale = 2.5
    for i, word in enumerate(words):
        word.x = center_x + coords[i, 0] * scale
        word.y = center_y + coords[i, 1] * scale
        word.rotation = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 90])

    # Step 2: 徐々に凝縮
    if verbose:
        print("  Step 2: 凝縮中...")

    def count_overlaps():
        count = 0
        for i, w1 in enumerate(words):
            for w2 in words[i+1:]:
                if w1.overlaps(w2, padding=2):
                    count += 1
        return count

    def resolve_overlaps_by_swap_or_nudge():
        """重なりを交換または微調整で解消"""
        for i, word in enumerate(words):
            for j, other in enumerate(words):
                if i >= j:
                    continue
                if word.overlaps(other, padding=2):
                    # 方法1: 位置交換を試す
                    old_x1, old_y1 = word.x, word.y
                    old_x2, old_y2 = other.x, other.y

                    word.x, word.y = old_x2, old_y2
                    other.x, other.y = old_x1, old_y1

                    # 交換後の重なりをチェック
                    new_overlaps_word = sum(1 for k, w in enumerate(words) if k != i and word.overlaps(w, padding=2))
                    new_overlaps_other = sum(1 for k, w in enumerate(words) if k != j and other.overlaps(w, padding=2))

                    # 交換で改善しなければ元に戻してnudge
                    if new_overlaps_word + new_overlaps_other >= 2:
                        word.x, word.y = old_x1, old_y1
                        other.x, other.y = old_x2, old_y2

                        # 方法2: 少しずらす（nudge）
                        dx = word.x - other.x
                        dy = word.y - other.y
                        dist = max(np.sqrt(dx*dx + dy*dy), 0.1)
                        nudge = 3.0
                        word.x += (dx / dist) * nudge
                        word.y += (dy / dist) * nudge
                        other.x -= (dx / dist) * nudge
                        other.y -= (dy / dist) * nudge

    best_scale = scale
    best_positions = [(w.x, w.y) for w in words]
    best_overlaps = count_overlaps()

    for iteration in range(iterations):
        # 縮小率（徐々に小さく）
        shrink_rate = 0.995

        # 全体を中心に向かって縮小
        for word in words:
            dx = word.x - center_x
            dy = word.y - center_y
            word.x = center_x + dx * shrink_rate
            word.y = center_y + dy * shrink_rate

        # 重なりを解消
        for _ in range(3):
            resolve_overlaps_by_swap_or_nudge()

        # 境界内に収める
        for word in words:
            margin = 20
            word.x = max(margin + word.width/2, min(canvas_width - margin - word.width/2, word.x))
            word.y = max(margin + word.height/2, min(canvas_height - margin - word.height/2, word.y))

        current_overlaps = count_overlaps()

        # 重なりが少ない状態を記録
        if current_overlaps <= best_overlaps:
            best_overlaps = current_overlaps
            best_positions = [(w.x, w.y) for w in words]

        if verbose and iteration % 30 == 0:
            print(f"    反復 {iteration}: 重なり={current_overlaps}")

        # 重なりがなくなったら終了
        if current_overlaps == 0:
            if verbose:
                print(f"    反復 {iteration} で重なり解消完了")
            break

    # 最良の状態に戻す
    for i, word in enumerate(words):
        word.x, word.y = best_positions[i]

    # 最終調整
    if verbose:
        print("  Step 3: 最終調整...")
    for _ in range(50):
        resolve_overlaps_by_swap_or_nudge()
        # 境界内に収める
        for word in words:
            margin = 20
            word.x = max(margin + word.width/2, min(canvas_width - margin - word.width/2, word.x))
            word.y = max(margin + word.height/2, min(canvas_height - margin - word.height/2, word.y))

    final_overlaps = count_overlaps()
    if verbose:
        print(f"  完了: 最終重なり={final_overlaps}")

    return words


def force_directed_layout(
    words: list[Word],
    similarity_matrix: np.ndarray,
    canvas_width: float = 800,
    canvas_height: float = 600,
    iterations: int = 200,
    verbose: bool = True
):
    """意味的配置＋凝縮レイアウト"""
    return semantic_compact_layout(
        words, similarity_matrix, canvas_width, canvas_height, iterations, verbose
    )


def compute_semantic_colors(words: list[Word]) -> list[tuple]:
    """埋め込みに基づいて色を計算（似た意味→似た色、色相環を循環）"""
    from sklearn.decomposition import PCA
    import colorsys

    # 埋め込みを取得
    embeddings = np.array([w.embedding for w in words])

    # PCAで2次元に次元削減
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)

    # 各点の角度を計算（色相環にマッピング）
    colors = []
    for x, y in coords_2d:
        # atan2で角度を計算（-π〜π）→ 0〜1に正規化
        angle = np.arctan2(y, x)  # -π〜π
        hue = (angle + np.pi) / (2 * np.pi)  # 0〜1（色相環全体）

        # 中心からの距離で彩度を調整
        dist = np.sqrt(x*x + y*y)
        max_dist = np.max(np.sqrt(coords_2d[:, 0]**2 + coords_2d[:, 1]**2))
        saturation = 0.6 + 0.35 * (dist / max_dist if max_dist > 0 else 0)

        value = 0.85  # 明度は固定

        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)

    return colors


def render_wordcloud(
    words: list[Word],
    similarity_matrix: np.ndarray,
    output_path: str,
    canvas_width: float = 800,
    canvas_height: float = 600
):
    """ワードクラウドを描画"""
    font_path = get_font_path()
    font_prop = font_manager.FontProperties(fname=font_path) if font_path else None

    fig, ax = plt.subplots(figsize=(canvas_width/80, canvas_height/80))
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_aspect('equal')
    ax.axis('off')

    # 意味的類似度に基づいて色を計算
    colors = compute_semantic_colors(words)

    for i, word in enumerate(words):
        color = colors[i]
        rotation = word.rotation

        ax.text(
            word.x, word.y, word.text,
            fontsize=word.font_size,
            fontproperties=font_prop,
            color=color,
            ha='center', va='center',
            rotation=rotation,
            alpha=0.9
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"保存しました: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Force-directed semantic wordcloud')
    parser.add_argument('input', help='入力ファイル')
    parser.add_argument('-o', '--output', default='semantic_wordcloud_v2.png')
    parser.add_argument('-n', '--num-words', type=int, default=80)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--api-key', help='OpenAI APIキー')
    args = parser.parse_args()

    # テキスト読み込み
    print(f"ファイル読み込み中: {args.input}")
    text = load_text(args.input)

    # 単語抽出
    extracted = extract_words(text)
    word_freq = Counter(extracted)

    print(f"抽出した単語数: {len(extracted)}")
    print(f"ユニークな単語数: {len(word_freq)}")

    # 上位N個
    top_words = [w for w, _ in word_freq.most_common(args.num_words)]
    top_freqs = {w: word_freq[w] for w in top_words}

    print(f"上位{len(top_words)}単語を使用")

    # 埋め込み取得
    embeddings = get_embeddings(top_words, args.api_key)

    # 類似度行列
    similarity_matrix = cosine_similarity(embeddings)

    # フォントサイズ計算
    max_freq = max(top_freqs.values())
    min_freq = min(top_freqs.values())

    words = []
    for i, w in enumerate(top_words):
        freq = top_freqs[w]
        # フォントサイズ: 12〜40
        if max_freq > min_freq:
            ratio = (freq - min_freq) / (max_freq - min_freq)
            font_size = 12 + 28 * (ratio ** 0.5)
        else:
            font_size = 20

        words.append(Word(w, freq, font_size, embeddings[i]))

    # Force-directed layout
    print("Force-directed layoutを実行中...")
    words = force_directed_layout(
        words, similarity_matrix,
        canvas_width=800, canvas_height=600,
        iterations=args.iterations
    )

    # 描画
    render_wordcloud(words, similarity_matrix, args.output)


if __name__ == '__main__':
    main()
