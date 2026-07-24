"""
Нормализация имён слоёв металлизации между разными PDK.

Разные PDK называют один и тот же тип слоя по-разному в DRC-репортах:
  met1 / M1 / metal1 / MET1  -> все это "первый металл"
  li1 / LI / LI1              -> локальный interconnect
  via1 / V1                   -> via-слой

Эта функция приводит их к единому канону "PREFIX+номер" (M1, LI1, V1, ...),
чтобы статистика и таргеты были сравнимы между PDK. Регэксп-эвристика
покрывает основные конвенции, но НЕ гарантированно покрывает все PDK —
после первого прогона на новом датасете стоит вызвать
`report_unmapped_layers()` и глазами проверить, что ничего не съехало
(особенно бинарные случаи вроде "LI" без номера, см. ниже).
"""

import re
from collections import Counter

# Известные семейства префиксов -> канонический короткий префикс.
# Ключи — в нижнем регистре, сравнение регистронезависимое.
_FAMILY_ALIASES = {
    "li": "LI",
    "local": "LI",
    "licon": "LI",
    "met": "M",
    "metal": "M",
    "m": "M",
    "via": "V",
    "v": "V",
    "mcon": "MCON",
}

# Если у семейства нет номера в исходном имени (например просто "LI"),
# по умолчанию считаем это первым слоем семейства. Это предположение может
# быть неверным для конкретного PDK — стоит проверить руками на новом PDK.
_DEFAULT_NUMBER_IF_MISSING = {"LI": "1"}

_LAYER_RE = re.compile(r"^([A-Za-z]+)[\s_]*?(\d*)$")

# Ручные overrides для конкретных строк, которые не укладываются в общую
# регэксп-логику (заполняется по мере обнаружения "непонятных" слоёв в
# новых PDK — см. report_unmapped_layers()).
LAYER_ALIAS_OVERRIDES: dict[str, str] = {
    # "somebizzarename": "M3",
}


def normalize_layer_name(raw: str) -> str:
    if raw is None:
        return raw
    raw = raw.strip()

    if raw in LAYER_ALIAS_OVERRIDES:
        return LAYER_ALIAS_OVERRIDES[raw]

    m = _LAYER_RE.match(raw)
    if not m:
        # Не похоже на "буквы+цифры" (например составное имя) — возвращаем
        # как есть в верхнем регистре, чтобы хотя бы не плодить дубликаты
        # из-за регистра, и оставляем видимым для report_unmapped_layers().
        return raw.upper()

    prefix, num = m.group(1).lower(), m.group(2)
    canonical_prefix = _FAMILY_ALIASES.get(prefix)
    if canonical_prefix is None:
        # Незнакомый префикс — не угадываем, оставляем как есть (в верхнем
        # регистре), чтобы это было видно в report_unmapped_layers() и можно
        # было добавить в _FAMILY_ALIASES или LAYER_ALIAS_OVERRIDES осознанно.
        return raw.upper()

    if not num:
        num = _DEFAULT_NUMBER_IF_MISSING.get(canonical_prefix, "")

    return f"{canonical_prefix}{num}"


def report_unmapped_layers(raw_layer_counts: Counter, top_n: int = 30):
    """Показывает, какие исходные имена слоёв на что нормализовались —
    прогнать один раз на новом PDK/датасете и глазами проверить, что
    маппинг адекватен, прежде чем доверять статистике/таргетам."""
    print("\n=== Проверка нормализации имён слоёв (raw -> normalized) ===")
    for raw, count in raw_layer_counts.most_common(top_n):
        norm = normalize_layer_name(raw)
        print(f"  {raw!r:15s} -> {norm!r:10s}  (count={count})")