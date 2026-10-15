"""Сколько меток test различается между сабмитами (только подсчёт, никаких статистик по снимкам test).

usage: python compare_submissions.py final_submission.csv ../candidates/A_2gpu_r50_r101_r152_inc3.csv [...]
Ожидание для нового прогона того же ансамбля A: примерно 40-150 различий с кандидатом A (шум перезапуска).
Если различий больше 300, вероятно, сломан порядок id или соответствие меток классам.
"""
import itertools
import sys

import pandas as pd

subs = {path: pd.read_csv(path, dtype={"id": str}) for path in sys.argv[1:]}
first = next(iter(subs.values()))
for path, s in subs.items():
    assert list(s.columns) == ["id", "label"], f"{path}: колонки {list(s.columns)}"
    assert s.id.tolist() == first.id.tolist(), f"{path}: другой порядок или набор id"
    assert len(s) == 8299 and s.label.between(0, 19).all(), f"{path}: {len(s)} строк или метки вне 0..19"
for a, b in itertools.combinations(subs, 2):
    n = int((subs[a].label.to_numpy() != subs[b].label.to_numpy()).sum())
    print(f"{n:5d} различий ({n / len(first):.2%}): {a}  vs  {b}")
