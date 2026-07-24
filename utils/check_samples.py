"""
Два количественных теста на выравнивание input<->target, без разглядывания картинок.

Тест A: сколько позитивных таргет-точек падают на RUDY==0 (то есть за пределами
реальной area, в padding/пустой зоне). Если сетки выровнены правильно, DRC
нарушение физически не может лежать в зоне, где вообще нет фичей (там нет
трассировки/размещения) -> доля должна быть близка к 0. Заметная доля > 0 -
сильный сигнал, что таргет и фичи не совпадают пространственно.

Тест B: систематический сдвиг. Смещаем target_objectness на (dx,dy) в диапазоне
[-3..3] и считаем средний RUDY под позитивными пикселями для каждого сдвига.
Если пик приходится не на (dx,dy)=(0,0), а стабильно на какой-то другой сдвиг
по многим сэмплам - это отпечаток off-by-one / неправильного origin где-то
в парсинге DEF/RUDY.
"""
import numpy as np
from pathlib import Path
import sys

def check_sample(npz_path):
    d = np.load(npz_path)
    rudy = d["rudy"]
    obj = d["target_objectness"]
    mask = d["target_offset_mask"]  # дискретные позитивные ячейки (лучше, чем thresholding по gaussian)

    assert rudy.shape == obj.shape == mask.shape, f"SHAPE MISMATCH: rudy={rudy.shape} obj={obj.shape} mask={mask.shape}"

    pos = mask > 0
    n_pos = pos.sum()
    if n_pos == 0:
        return None

    # Тест A
    on_zero_rudy = (rudy[pos] == 0).sum()
    frac_on_zero = on_zero_rudy / n_pos

    # Тест B: перебор сдвигов
    best_shift, best_score = (0, 0), -1
    scores = {}
    ys, xs = np.where(pos)
    H, W = rudy.shape
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            yy = ys + dy
            xx = xs + dx
            valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            if valid.sum() == 0:
                continue
            score = rudy[yy[valid], xx[valid]].mean()
            scores[(dy, dx)] = score
            if score > best_score:
                best_score, best_shift = score, (dy, dx)

    return {
        "n_pos": int(n_pos),
        "frac_on_zero_rudy": float(frac_on_zero),
        "best_shift_dy_dx": best_shift,
        "score_at_00": scores.get((0, 0), None),
        "best_score": best_score,
    }

if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    files = sorted(data_dir.glob("*.npz"))
    print(f"Найдено файлов: {len(files)}")

    shift_votes = {}
    frac_list = []
    for f in files:
        try:
            res = check_sample(f)
        except AssertionError as e:
            print(f"[SHAPE BUG] {f.name}: {e}")
            continue
        if res is None:
            continue
        frac_list.append(res["frac_on_zero_rudy"])
        shift_votes[res["best_shift_dy_dx"]] = shift_votes.get(res["best_shift_dy_dx"], 0) + 1
        print(f"{f.name}: n_pos={res['n_pos']:5d}  frac_on_zero_rudy={res['frac_on_zero_rudy']:.3f}  "
              f"best_shift(dy,dx)={res['best_shift_dy_dx']}  score@(0,0)={res['score_at_00']:.3f}  "
              f"best_score={res['best_score']:.3f}")

    if frac_list:
        print(f"\n=== ИТОГО по {len(frac_list)} сэмплам ===")
        print(f"Средняя доля таргетов на RUDY==0: {np.mean(frac_list):.3f} (должна быть близка к 0)")
        print(f"Голосование за лучший сдвиг (dy,dx): {sorted(shift_votes.items(), key=lambda x: -x[1])}")
        print("Если (0,0) не побеждает уверенно по большинству сэмплов - похоже на систематический сдвиг сетки.")