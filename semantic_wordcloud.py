#!/usr/bin/env python3
"""v6: キーワードのみの意味的ワードクラウド（高次元距離制約）"""

import argparse
import os
import random
import json
import hashlib
import urllib.request
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from openai import OpenAI
from janome.tokenizer import Tokenizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import font_manager
import colorsys


FONT_PATHS = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
]

# 日本語ストップワード（stopwords-iso）
JAPANESE_STOPWORDS_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ja/master/stopwords-ja.txt"
# 英語ストップワード（stopwords-iso）
ENGLISH_STOPWORDS_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt"
STOPWORDS_CACHE_FILE = "stopwords_cache.json"


def load_stopwords(custom_filepath: str = "stopwords.txt") -> set:
    """ストップワードを読み込む（NLTK英語 + SlothLib日本語 + カスタム）"""
    stopwords = set()

    # キャッシュから読み込み試行
    cache_path = Path(STOPWORDS_CACHE_FILE)
    cached_standard = None
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_standard = set(json.load(f))
            print(f"標準ストップワードをキャッシュから読み込みました（{len(cached_standard)}語）")
        except Exception:
            cached_standard = None

    if cached_standard:
        stopwords.update(cached_standard)
    else:
        # 日本語ストップワード（stopwords-iso）
        try:
            with urllib.request.urlopen(JAPANESE_STOPWORDS_URL, timeout=10) as response:
                ja_words = response.read().decode('utf-8').split('\n')
                ja_words = [w.strip() for w in ja_words if w.strip()]
                stopwords.update(ja_words)
                print(f"日本語ストップワード: {len(ja_words)}語")
        except Exception as e:
            print(f"日本語ストップワード読み込みエラー: {e}")

        # 英語ストップワード（stopwords-iso）
        try:
            with urllib.request.urlopen(ENGLISH_STOPWORDS_URL, timeout=10) as response:
                en_words = response.read().decode('utf-8').split('\n')
                en_words = [w.strip() for w in en_words if w.strip()]
                stopwords.update(en_words)
                print(f"英語ストップワード: {len(en_words)}語")
        except Exception as e:
            print(f"英語ストップワード読み込みエラー: {e}")

        # 標準ストップワードをキャッシュに保存
        if stopwords:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(list(stopwords), f, ensure_ascii=False)
            print(f"標準ストップワードをキャッシュに保存しました")

    # カスタムストップワード（ローカルファイル）
    custom_path = Path(custom_filepath)
    if custom_path.exists():
        custom_count = 0
        with open(custom_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    stopwords.add(line.lower())
                    stopwords.add(line)
                    custom_count += 1
        print(f"カスタムストップワード: {custom_count}語")
    else:
        print(f"警告: カスタムストップワードファイルが見つかりません: {custom_filepath}")

    print(f"合計ストップワード: {len(stopwords)}語")
    return stopwords


STOPWORDS = load_stopwords()


class Word:
    def __init__(self, text: str, freq: int, font_size: float, embedding: np.ndarray):
        self.text = text
        self.freq = freq
        self.font_size = font_size
        self.embedding = embedding
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.rotation = 0

    @property
    def width(self) -> float:
        char_width = self.font_size * 1.0
        if self.rotation == 0:
            return len(self.text) * char_width
        else:
            return self.font_size * 1.2

    @property
    def height(self) -> float:
        if self.rotation == 0:
            return self.font_size * 1.2
        else:
            return len(self.text) * self.font_size * 1.0

    def get_bbox(self) -> tuple:
        half_w = self.width / 2
        half_h = self.height / 2
        return (self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h)

    def overlaps(self, other: 'Word', padding: float = 2.0) -> bool:
        b1 = self.get_bbox()
        b2 = other.get_bbox()
        return not (b1[2] + padding < b2[0] or b2[2] + padding < b1[0] or
                    b1[3] + padding < b2[1] or b2[3] + padding < b1[1])


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


def load_custom_words(file_path: str) -> dict[str, float]:
    """カスタム単語ファイルを読み込む（タブ区切り: 単語\t正規化値）

    正規化値は0.0〜1.0の範囲で、文字サイズに直接対応:
      0.0 = 最小サイズ（12pt）
      1.0 = 最大サイズ（40pt）
    """
    custom_words = {}
    path = Path(file_path)
    if not path.exists():
        print(f"警告: カスタム単語ファイルが見つかりません: {file_path}")
        return custom_words

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                word = parts[0].strip()
                try:
                    ratio = float(parts[1].strip())
                    if not 0.0 <= ratio <= 1.0:
                        print(f"警告: 行{line_num}の値が範囲外です（0.0-1.0）: {ratio}")
                        ratio = max(0.0, min(1.0, ratio))
                    custom_words[word] = ratio
                except ValueError:
                    print(f"警告: 行{line_num}の値が不正です: {line}")
            elif len(parts) == 1:
                # 値省略時はデフォルト（中間サイズ）
                custom_words[parts[0].strip()] = 0.5
                print(f"情報: 行{line_num}の値をデフォルト(0.5)に設定: {parts[0].strip()}")

    print(f"カスタム単語: {len(custom_words)}語")
    return custom_words


def get_embeddings(words: list[str], api_key: str = None, cache_file: str = "embeddings_cache.json") -> np.ndarray:
    """埋め込みを取得（単語単位キャッシュ対応）"""

    # キャッシュ読み込み
    cache = {}
    if Path(cache_file).exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception as e:
            print(f"キャッシュ読み込みエラー: {e}")
            cache = {}

    # キャッシュにある単語とない単語を分離
    cached_words = [w for w in words if w in cache]
    new_words = [w for w in words if w not in cache]

    if cached_words:
        print(f"キャッシュから{len(cached_words)}個の埋め込みを読み込みました")

    # 新しい単語のみAPIで取得
    if new_words:
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        print(f"OpenAI APIで{len(new_words)}個の単語の埋め込みを取得中...")
        batch_size = 100
        for i in range(0, len(new_words), batch_size):
            batch = new_words[i:i+batch_size]
            response = client.embeddings.create(model="text-embedding-3-small", input=batch)
            for word, item in zip(batch, response.data):
                cache[word] = item.embedding
            print(f"  {min(i+batch_size, len(new_words))}/{len(new_words)} 完了")

        # キャッシュ保存
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"キャッシュを保存しました: {cache_file}（合計{len(cache)}単語）")

    # 要求された順序で埋め込みを返す
    embeddings = [cache[w] for w in words]
    return np.array(embeddings)


def force_directed_layout(
    words: list[Word],
    canvas_width: float = 800,
    canvas_height: float = 600,
    iterations: int = 500,
    verbose: bool = True
):
    """高次元距離を制約として使用するForce-directed layout（異方性対応）"""

    n = len(words)
    center_x = canvas_width / 2
    center_y = canvas_height / 2

    # アスペクト比に基づく異方性スケール
    aspect_ratio = canvas_width / canvas_height
    scale_x = np.sqrt(aspect_ratio)  # 横方向に伸ばす
    scale_y = 1.0 / np.sqrt(aspect_ratio)  # 縦方向に縮める

    # 高次元での距離行列を計算
    if verbose:
        print("  高次元距離行列を計算...")
    embeddings = np.array([w.embedding for w in words])
    sim_matrix = cosine_similarity(embeddings)
    dist_matrix = 1 - sim_matrix

    max_dist = np.max(dist_matrix)
    ideal_scale = min(canvas_width, canvas_height) * 0.35
    ideal_distances = (dist_matrix / max_dist) * ideal_scale

    # PCAで初期配置（異方性スケール適用）
    if verbose:
        print("  PCAで初期配置...")
    pca = PCA(n_components=2)
    initial_coords = pca.fit_transform(embeddings)
    initial_coords[:, 0] -= initial_coords[:, 0].mean()
    initial_coords[:, 1] -= initial_coords[:, 1].mean()

    scale = ideal_scale / (np.max(np.abs(initial_coords)) + 1e-6)
    for i, word in enumerate(words):
        word.x = center_x + initial_coords[i, 0] * scale * scale_x
        word.y = center_y + initial_coords[i, 1] * scale * scale_y
        word.vx = 0
        word.vy = 0
        word.rotation = random.choice([0]*9 + [90])

    # Force-directed simulation
    if verbose:
        print("  Force-directed simulation...")

    attraction_strength = 0.01
    repulsion_strength = 500
    damping = 0.9

    # 反発力計算用の固定サイズ（配置を意味のみで決定するため）
    fixed_word_size = 50  # 全単語を同じサイズとして扱う

    def compute_forces():
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                dx = words[j].x - words[i].x
                dy = words[j].y - words[i].y
                # 異方性距離（楕円距離）
                normalized_dx = dx / scale_x
                normalized_dy = dy / scale_y
                current_dist = np.sqrt(normalized_dx**2 + normalized_dy**2) + 1e-6
                ideal_dist = ideal_distances[i, j]
                ux, uy = normalized_dx / current_dist, normalized_dy / current_dist

                spring_force = attraction_strength * (current_dist - ideal_dist)
                # 異方性を適用した力
                forces[i, 0] += spring_force * ux * scale_x
                forces[i, 1] += spring_force * uy * scale_y
                forces[j, 0] -= spring_force * ux * scale_x
                forces[j, 1] -= spring_force * uy * scale_y

                # 反発力は固定サイズで計算（フォントサイズに依存しない）
                min_sep = fixed_word_size + 5
                actual_dist = np.sqrt(dx*dx + dy*dy) + 1e-6
                if actual_dist < min_sep:
                    repulsion = repulsion_strength / (actual_dist * actual_dist + 1)
                    forces[i, 0] -= repulsion * dx / actual_dist
                    forces[i, 1] -= repulsion * dy / actual_dist
                    forces[j, 0] += repulsion * dx / actual_dist
                    forces[j, 1] += repulsion * dy / actual_dist

        # 中心への引力（異方性：楕円の中心に向かう）
        for i in range(n):
            dx = center_x - words[i].x
            dy = center_y - words[i].y
            # 楕円距離で正規化
            normalized_dx = dx / scale_x
            normalized_dy = dy / scale_y
            dist_to_center = np.sqrt(normalized_dx**2 + normalized_dy**2) + 1e-6
            center_force = 0.002 * dist_to_center
            forces[i, 0] += center_force * normalized_dx / dist_to_center * scale_x
            forces[i, 1] += center_force * normalized_dy / dist_to_center * scale_y

        return forces

    def count_overlaps():
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if words[i].overlaps(words[j], padding=2):
                    count += 1
        return count

    best_positions = [(w.x, w.y) for w in words]
    best_overlaps = count_overlaps()

    for iteration in range(iterations):
        forces = compute_forces()
        temperature = max(0.1, 1.0 - iteration / iterations)

        for i, word in enumerate(words):
            word.vx = (word.vx + forces[i, 0]) * damping * temperature
            word.vy = (word.vy + forces[i, 1]) * damping * temperature

            speed = np.sqrt(word.vx**2 + word.vy**2)
            max_speed = 10 * temperature
            if speed > max_speed:
                word.vx = word.vx / speed * max_speed
                word.vy = word.vy / speed * max_speed

            word.x += word.vx
            word.y += word.vy

            margin = 30
            word.x = max(margin + word.width/2, min(canvas_width - margin - word.width/2, word.x))
            word.y = max(margin + word.height/2, min(canvas_height - margin - word.height/2, word.y))

        current_overlaps = count_overlaps()
        if current_overlaps <= best_overlaps:
            best_overlaps = current_overlaps
            best_positions = [(w.x, w.y) for w in words]

        if verbose and iteration % 100 == 0:
            print(f"    反復 {iteration}: 重なり={current_overlaps}")

    for i, word in enumerate(words):
        word.x, word.y = best_positions[i]

    # 最終重なり解消
    for _ in range(100):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if words[i].overlaps(words[j], padding=2):
                    dx = words[i].x - words[j].x
                    dy = words[i].y - words[j].y
                    dist = max(np.sqrt(dx*dx + dy*dy), 0.1)
                    nudge = 2.0
                    words[i].x += (dx / dist) * nudge
                    words[i].y += (dy / dist) * nudge
                    words[j].x -= (dx / dist) * nudge
                    words[j].y -= (dy / dist) * nudge
                    improved = True

        for word in words:
            margin = 30
            word.x = max(margin + word.width/2, min(canvas_width - margin - word.width/2, word.x))
            word.y = max(margin + word.height/2, min(canvas_height - margin - word.height/2, word.y))

        if not improved:
            break

    if verbose:
        print(f"  完了: 最終重なり={count_overlaps()}")

    return words


def compute_semantic_colors(words: list[Word]) -> list[tuple]:
    """埋め込みに基づいて色を計算（色相環）"""
    embeddings = np.array([w.embedding for w in words])
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)

    colors = []
    max_dist = np.max(np.sqrt(coords_2d[:, 0]**2 + coords_2d[:, 1]**2))
    for x, y in coords_2d:
        angle = np.arctan2(y, x)
        hue = (angle + np.pi) / (2 * np.pi)
        dist = np.sqrt(x*x + y*y)
        saturation = 0.65 + 0.3 * min(dist / max_dist, 1.0)
        value = 0.85
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors


def render_wordcloud(words: list[Word], output_path: str, canvas_width: float = 800, canvas_height: float = 600):
    font_path = get_font_path()
    font_prop = font_manager.FontProperties(fname=font_path) if font_path else None

    fig, ax = plt.subplots(figsize=(canvas_width/80, canvas_height/80))
    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_aspect('equal')
    ax.axis('off')

    colors = compute_semantic_colors(words)

    for i, word in enumerate(words):
        ax.text(word.x, word.y, word.text, fontsize=word.font_size,
                fontproperties=font_prop, color=colors[i],
                ha='center', va='center', rotation=word.rotation, alpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"保存しました: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='意味的ワードクラウド（キーワードのみ）')
    parser.add_argument('input', nargs='?', help='入力ファイル（テキスト/Excel）')
    parser.add_argument('-o', '--output', default='semantic_wordcloud.png')
    parser.add_argument('-n', '--num-words', type=int, default=80)
    parser.add_argument('--custom-words', help='カスタム単語ファイル（タブ区切り: 単語\\t頻度）')
    parser.add_argument('--export-words', help='選択された単語をCSVに出力')
    parser.add_argument('--cache-words', type=int, default=200, help='キャッシュする単語数（表示数より多めに）')
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--seed', type=int, default=None, help='ランダムシード（再現性確保用）')
    parser.add_argument('--api-key', help='OpenAI APIキー')
    args = parser.parse_args()

    if not args.input and not args.custom_words:
        parser.error('入力ファイルまたは--custom-wordsを指定してください')

    if args.seed is not None:
        random.seed(args.seed)
        print(f"ランダムシード: {args.seed}")

    # テキストから単語抽出
    word_freq = Counter()
    if args.input:
        print(f"ファイル読み込み中: {args.input}")
        text = load_text(args.input)
        extracted = extract_words(text)
        word_freq = Counter(extracted)
        print(f"抽出: {len(extracted)}語, ユニーク: {len(word_freq)}語")

    # カスタム単語を読み込み（正規化値を保持）
    custom_ratios = {}
    if args.custom_words:
        custom_ratios = load_custom_words(args.custom_words)
        for word in custom_ratios:
            if word not in word_freq:
                word_freq[word] = 0  # テキストに出現しない場合は0

    if not word_freq:
        print("エラー: 単語が見つかりません")
        return

    # キャッシュ用に多めに取得
    cache_words = [w for w, _ in word_freq.most_common(args.cache_words)]
    if len(cache_words) > args.num_words:
        print(f"キャッシュ用に上位{len(cache_words)}単語の埋め込みを取得")
        get_embeddings(cache_words, args.api_key)

    # 表示用単語を選択（カスタム単語は必ず含める）
    # カスタム単語以外から上位N語を選択
    num_from_extracted = args.num_words - len(custom_ratios)
    extracted_words = [w for w, _ in word_freq.most_common() if w not in custom_ratios][:max(0, num_from_extracted)]
    top_words = list(custom_ratios.keys()) + extracted_words
    top_freqs = {w: word_freq[w] for w in top_words}

    print(f"表示: {len(top_words)}単語（カスタム: {len(custom_ratios)}語）")

    embeddings = get_embeddings(top_words, args.api_key)

    # 抽出単語のみで正規化範囲を計算
    extracted_freqs = [word_freq[w] for w in extracted_words] if extracted_words else [1]
    max_freq = max(extracted_freqs)
    min_freq = min(extracted_freqs)

    words = []
    word_ratios = {}  # CSV出力用に保存
    for i, w in enumerate(top_words):
        freq = top_freqs[w]
        if w in custom_ratios:
            # カスタム単語: 正規化値を直接使用
            ratio = custom_ratios[w]
        else:
            # 抽出単語: 頻度から正規化
            if max_freq > min_freq:
                ratio = (freq - min_freq) / (max_freq - min_freq)
            else:
                ratio = 0.5
        font_size = 12 + 28 * (ratio ** 0.5)
        word_ratios[w] = ratio
        words.append(Word(w, freq, font_size, embeddings[i]))

    # 単語リストをCSV出力
    if args.export_words:
        import csv
        # 正規化値で降順ソート
        sorted_words = sorted(top_words, key=lambda w: word_ratios[w], reverse=True)
        with open(args.export_words, 'w', encoding='utf-8-sig', newline='') as f:  # BOM付きUTF-8
            writer = csv.writer(f)
            writer.writerow(['単語', '出現回数', '正規化値', 'フォントサイズ', 'カスタム', '頻度計算値', '差分'])
            for w in sorted_words:
                freq = top_freqs[w]
                ratio = word_ratios[w]
                font_size = 12 + 28 * (ratio ** 0.5)
                is_custom = 'Yes' if w in custom_ratios else ''
                # 頻度から計算した場合の値
                if max_freq > min_freq:
                    freq_ratio = (freq - min_freq) / (max_freq - min_freq)
                else:
                    freq_ratio = 0.5
                # 差分（カスタム値 - 頻度計算値）
                diff = ratio - freq_ratio if w in custom_ratios else ''
                freq_ratio_str = f'{freq_ratio:.3f}'
                diff_str = f'{diff:+.3f}' if diff != '' else ''
                writer.writerow([w, freq, f'{ratio:.3f}', f'{font_size:.1f}', is_custom, freq_ratio_str, diff_str])
        print(f"単語リストを出力しました: {args.export_words}")

    print("レイアウト実行中...")
    words = force_directed_layout(words, canvas_width=900, canvas_height=700, iterations=args.iterations)

    render_wordcloud(words, args.output, canvas_width=900, canvas_height=700)


if __name__ == '__main__':
    main()
