"""
sample_generator.py
実験・検証用のサンプルカード（マスター画像）と、
それを机の上に置いて斜め撮影したような「擬似写真テストデータ」を自動生成するスクリプト
"""

import cv2
import numpy as np
import math
from pathlib import Path
import random
import sys

# Windowsコンソール文字コード対応
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def draw_card_template(
    card_id: str,
    name: str,
    theme_color: tuple,
    pattern_type: str,
    card_type: str = "MONSTER",
    attack: int = 2500,
    defense: int = 2000,
    width: int = 600,
    height: int = 840,
) -> np.ndarray:
    """
    リッチなTCG風カード画像を合成して生成
    """
    card = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. 外枠と背景ベース
    cv2.rectangle(card, (0, 0), (width, height), theme_color, -1)
    # 内側フレーム
    cv2.rectangle(card, (25, 25), (width - 25, height - 25), (30, 30, 30), -1)
    cv2.rectangle(card, (35, 35), (width - 35, height - 35), (50, 50, 50), 3)

    # 2. タイトルヘッダー領域
    cv2.rectangle(card, (45, 45), (width - 45, 105), theme_color, -1)
    cv2.rectangle(card, (45, 45), (width - 45, 105), (255, 255, 255), 2)
    cv2.putText(
        card, name, (60, 85), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2
    )

    # 属性アイコン風サークル
    cv2.circle(card, (width - 75, 75), 20, (255, 255, 255), -1)
    cv2.circle(card, (width - 75, 75), 18, theme_color, -1)
    cv2.putText(
        card, card_type[0], (width - 83, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    # 3. メインイラスト領域 (45, 120) から (width-45, 520)
    ill_x1, ill_y1, ill_x2, ill_y2 = 45, 120, width - 45, 520
    ill_w, ill_h = ill_x2 - ill_x1, ill_y2 - ill_y1
    illustration = np.zeros((ill_h, ill_w, 3), dtype=np.uint8)

    # パターンに応じた幾何学・アート風イラストを描画
    center_x, center_y = ill_w // 2, ill_h // 2
    if pattern_type == "dragon":
        # 炎とドラゴンのシルエット風
        for r in range(150, 10, -15):
            c = (int(theme_color[0] * r / 150), int(theme_color[1] * r / 150), int(255 * r / 150))
            cv2.circle(illustration, (center_x, center_y), r, c, -1)
        # 翼と角の多角形
        pts1 = np.array([[center_x, center_y - 120], [center_x - 180, center_y - 60], [center_x - 60, center_y + 40]])
        pts2 = np.array([[center_x, center_y - 120], [center_x + 180, center_y - 60], [center_x + 60, center_y + 40]])
        cv2.fillPoly(illustration, [pts1, pts2], (240, 240, 255))
        cv2.circle(illustration, (center_x, center_y - 20), 45, (40, 40, 200), -1)
    elif pattern_type == "cyber_mech":
        # サイバー・メカニック風グリッドと六角形
        for i in range(0, ill_w, 30):
            cv2.line(illustration, (i, 0), (i, ill_h), (60, 100, 60), 1)
        for j in range(0, ill_h, 30):
            cv2.line(illustration, (0, j), (ill_w, j), (60, 100, 60), 1)
        # 巨大六角形コア
        hex_pts = []
        for a in range(0, 360, 60):
            rad = math.radians(a)
            hx = int(center_x + 110 * math.cos(rad))
            hy = int(center_y + 110 * math.sin(rad))
            hex_pts.append([hx, hy])
        cv2.fillPoly(illustration, [np.array(hex_pts)], (200, 220, 50))
        cv2.circle(illustration, (center_x, center_y), 50, (255, 255, 255), -1)
        cv2.circle(illustration, (center_x, center_y), 30, (50, 150, 255), -1)
    elif pattern_type == "magic_sorcerer":
        # 魔法陣と星
        cv2.circle(illustration, (center_x, center_y), 130, (200, 100, 255), 3)
        cv2.circle(illustration, (center_x, center_y), 100, (150, 80, 200), 2)
        for a in range(0, 360, 45):
            rad = math.radians(a)
            cv2.line(
                illustration,
                (center_x, center_y),
                (int(center_x + 130 * math.cos(rad)), int(center_y + 130 * math.sin(rad))),
                (220, 180, 255),
                2,
            )
        cv2.circle(illustration, (center_x, center_y), 60, (255, 255, 200), -1)
    elif pattern_type == "phoenix":
        # 放射状の炎
        for a in range(0, 360, 15):
            rad = math.radians(a)
            p_end = (int(center_x + 160 * math.cos(rad)), int(center_y + 160 * math.sin(rad)))
            cv2.line(illustration, (center_x, center_y), p_end, (50, 180, 255), 4)
        cv2.circle(illustration, (center_x, center_y), 70, (0, 100, 255), -1)
        cv2.circle(illustration, (center_x, center_y), 40, (255, 255, 255), -1)
    else:
        # クリスタル・ダイヤモンド
        diamond_pts = np.array([
            [center_x, center_y - 140],
            [center_x + 120, center_y],
            [center_x, center_y + 140],
            [center_x - 120, center_y],
        ])
        cv2.fillPoly(illustration, [diamond_pts], (255, 230, 180))
        cv2.polylines(illustration, [diamond_pts], True, (255, 255, 255), 4)
        cv2.circle(illustration, (center_x, center_y), 50, (255, 100, 100), -1)

    card[ill_y1:ill_y2, ill_x1:ill_x2] = illustration
    cv2.rectangle(card, (ill_x1, ill_y1), (ill_x2, ill_y2), (220, 220, 220), 3)

    # 4. 説明テキストボックス
    cv2.rectangle(card, (45, 540), (width - 45, 740), (20, 20, 20), -1)
    cv2.rectangle(card, (45, 540), (width - 45, 740), theme_color, 2)
    cv2.putText(
        card, f"[{card_type} / EFFECT]", (60, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
    )
    cv2.putText(
        card, f"Card ID: {card_id}", (60, 605), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1
    )
    cv2.putText(
        card, "When this card is summoned, draw 1 card.", (60, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
    )
    cv2.putText(
        card, "Scale-invariant matching enabled.", (60, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
    )

    # 攻撃力 / 守備力
    cv2.line(card, (45, 700), (width - 45, 700), (80, 80, 80), 1)
    cv2.putText(
        card, f"ATK/ {attack}   DEF/ {defense}", (width - 270, 728), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1
    )

    # 下部カード番号
    cv2.putText(
        card, f"TCG-2026-{card_id.upper()}", (width - 230, 810), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
    )

    return card


def create_realistic_photo(
    card_img: np.ndarray,
    output_size: tuple = (1200, 900),
    angle_deg: float = 18.0,
    scale: float = 0.7,
) -> np.ndarray:
    """
    カード画像を木目風背景の上に斜め・パース変形させて配置し、スマホで撮影したようなリアルな写真を合成
    """
    out_w, out_h = output_size
    # 木目風のテクスチャ背景を作成
    bg = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for y in range(out_h):
        # 木目グラデーション
        wood_val = int(80 + 35 * math.sin(y / 25.0) + (y % 15) * 2)
        bg[y, :] = (wood_val // 2, wood_val * 2 // 3, wood_val)

    # カードの4隅の元座標
    ch, cw = card_img.shape[:2]
    src_pts = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)

    # 斜め撮影の4隅投影先座標（パース歪みと回転を付与）
    cx, cy = out_w // 2 + random.randint(-40, 40), out_h // 2 + random.randint(-30, 30)
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    # 歪みを持たせた投影先
    w_half = (cw * scale) / 2
    h_half = (ch * scale) / 2

    # パースペクティブ（上側を少し小さく、下側を大きくして奥行き表現）
    p_top = 0.85
    p_bot = 1.15

    corners = [
        (-w_half * p_top, -h_half),
        (w_half * p_top, -h_half),
        (w_half * p_bot, h_half),
        (-w_half * p_bot, h_half),
    ]

    dst_pts = []
    for x, y in corners:
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        dst_pts.append([rx, ry])
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # 射影変換行列を計算してカードを変形
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_card = cv2.warpPerspective(card_img, matrix, (out_w, out_h))

    # カードのマスク
    card_mask = np.zeros((ch, cw), dtype=np.uint8)
    card_mask[:, :] = 255
    warped_mask = cv2.warpPerspective(card_mask, matrix, (out_w, out_h))

    # 影の生成
    shadow_mask = cv2.GaussianBlur(warped_mask, (31, 31), 0)
    for c in range(3):
        bg[:, :, c] = np.where(
            warped_mask == 0,
            (bg[:, :, c] * (1.0 - 0.4 * (shadow_mask / 255.0))).astype(np.uint8),
            warped_card[:, :, c],
        )

    # 照明ムラ（グラデーション照明効果）
    light_map = np.zeros((out_h, out_w), dtype=np.float32)
    lx, ly = out_w // 4, out_h // 4
    for y in range(out_h):
        for x in range(0, out_w, 4):
            dist = math.sqrt((x - lx) ** 2 + (y - ly) ** 2)
            light_map[y, x : x + 4] = max(0.7, 1.25 - dist / 1200.0)

    for c in range(3):
        bg[:, :, c] = np.clip(bg[:, :, c] * light_map, 0, 255).astype(np.uint8)

    return bg


def generate_sample_dataset(master_dir: Path, test_dir: Path):
    """
    検証用のカードマスター10種と、対応するテスト用撮影写真（角度・背景変化）を生成
    """
    master_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    card_definitions = [
        ("001_Flame_Dragon", "Flame Dragon", (30, 40, 180), "dragon", "MONSTER", 2800, 2100),
        ("002_Cyber_Paladin", "Cyber Paladin", (180, 140, 30), "cyber_mech", "MACHINE", 2400, 1800),
        ("003_Arcane_Mage", "Arcane Mage", (180, 40, 140), "magic_sorcerer", "SPELLCASTER", 2100, 2500),
        ("004_Solar_Phoenix", "Solar Phoenix", (30, 120, 220), "phoenix", "BEAST", 2900, 2000),
        ("005_Crystal_Golem", "Crystal Golem", (180, 180, 50), "crystal", "ROCK", 1200, 3000),
        ("006_Inferno_Wyrm", "Inferno Wyrm", (20, 30, 210), "dragon", "DRAGON", 3000, 2500),
        ("007_Vector_Unit_01", "Vector Unit 01", (200, 120, 20), "cyber_mech", "MACHINE", 1900, 1500),
        ("008_Cosmic_Oracle", "Cosmic Oracle", (160, 50, 170), "magic_sorcerer", "FAIRY", 1700, 2600),
        ("009_Emerald_Bird", "Emerald Bird", (50, 180, 100), "phoenix", "WINGED", 2200, 1900),
        ("010_Prism_Guardian", "Prism Guardian", (210, 160, 80), "crystal", "WARRIOR", 2600, 2200),
    ]

    print("--- サンプルカードマスター画像を生成中 ---")
    for cid, name, color, ptype, ctype, atk, df in card_definitions:
        master_img = draw_card_template(cid, name, color, ptype, ctype, atk, df)
        save_path = master_dir / f"{cid}.png"
        cv2.imwrite(str(save_path), master_img)
        print(f"  作成: {save_path.name}")

    print("\n--- 擬似写真テストデータを生成中（斜め撮影・回転・影・照明効果） ---")
    # 各カードについて斜め撮影風テスト写真を生成
    for idx, (cid, name, color, ptype, ctype, atk, df) in enumerate(card_definitions, 1):
        master_img = cv2.imread(str(master_dir / f"{cid}.png"))
        angle = random.choice([-25.0, -15.0, 12.0, 22.0, 30.0])
        photo = create_realistic_photo(master_img, angle_deg=angle, scale=0.72)
        test_path = test_dir / f"test_photo_{cid}.jpg"
        cv2.imwrite(str(test_path), photo)
        print(f"  作成: {test_path.name} (傾き: {angle:.1f}度)")

    print(f"\n[OK] 完了: マスター画像10枚 ({master_dir}), テスト写真10枚 ({test_dir})")


if __name__ == "__main__":
    base = Path(__file__).parent / "data"
    generate_sample_dataset(base / "master", base / "test")
