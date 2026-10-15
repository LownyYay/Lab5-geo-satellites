"""Собирает Kaggle_45min.ipynb: финальное решение с обучением <= 45 мин на Kaggle (GPU T4 x2).
Код обучения берётся из worker.py (лежит рядом) и вставляется в ноутбук ячейкой %%writefile.

usage: python build_kaggle45.py [out.ipynb]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "Kaggle_45min.ipynb"
WORKER = (HERE / "worker.py").read_text(encoding="utf-8")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md(r"""
# Week 5, команда «Кудасики»: финальное решение с обучением не дольше 45 минут

Ноутбук **заново обучает** финальный ансамбль из четырёх сетей на всех размеченных снимках (train + val,
5698 штук) и пишет `final_submission.csv`. Время обучения измеряется `time.time()` и печатается крупно в
разделе 4. Лимит курса — 45 минут; мы целимся в 30–35 минут, чтобы был запас.

| GPU | модели, по очереди | вход, px | тест (TTA ×8), px | минут на Colab T4 (20 эпох) |
|---|---|---|---|---|
| 0 | ResNet-152 → ResNet-50 | 256 | 320 | 21.4 + 10.0 = 31.4 |
| 1 | ResNet-101 → Inception v3 | 256 / 299 | 320 / 363 | 15.2 + 12.8 = 28.1 |

Это **тот же ансамбль**, что дал 0.96678 на публичном лидерборде (val 0.9785, 15 ошибок из 699). Раньше модели
обучались по очереди на одной T4 — 59 минут. Здесь две видеокарты работают **одновременно**, каждая
обучает две модели, поэтому общее время — время более загруженной GPU, около 31–33 минут.

**Как запустить на Kaggle.**
1. *Add Input* → данные соревнования `week-5-competition-geo-satellites-classification`.
2. *Settings → Accelerator → GPU T4 x2*. Не P100: установленная версия torch собрана без поддержки P100
   (раздел 2 это проверяет).
3. *Settings → Internet → On*: нужен, чтобы скачать веса ImageNet из torchvision. Требует подтверждённого
   телефона в аккаунте Kaggle.
4. *Save Version → Save & Run All (Commit)*. Ноутбук выполнится в фоне, лог с временем сохранится в версии,
   а `final_submission.csv` появится в *Output*.
""")

md(r"""
## Почему решение устроено именно так

* **Модели и рецепт не меняли.** Всё проверено в holdout-прогонах (`Team_final.ipynb`): ResNet-50/101/152
  (`IMAGENET1K_V2`) и Inception v3, обучение всей сети, AdamW, разогрев + косинус, label smoothing,
  аугментация искажениями `RandomCorruption`, 20 эпох. Размер тестового снимка (320 / 363 px) выбран на val
  ещё тогда и здесь зафиксирован.
* **Почему 4 модели, а не больше или меньше.** Ансамбль из 2 моделей (r50 + r101) дал 0.96506, из 4 — 0.96678,
  из 7 — тоже 0.96678. Модели ошибаются почти на одних и тех же снимках, поэтому после 4 моделей выигрыша
  нет, а время растёт.
* **Почему два процесса, а не `DataParallel`.** Каждая GPU обучает свои модели независимо: рецепт не
  меняется (тот же батч 32, та же BatchNorm), ускорение почти ровно ×2. `DataParallel` делит один батч
  на две GPU и упирается в один процесс Python: ускорение заметно меньше ×2, а BatchNorm видит по 16 снимков.
* **Ускорения без изменения рецепта:** `cudnn.benchmark` (cuDNN подбирает самые быстрые свёртки), `AdamW(fused=True)`,
  счётчики loss / accuracy на GPU без синхронизации на каждом шаге.
* **Предохранитель времени.** Первая эпоха — разогрев lr, и её ход не зависит от общего числа эпох. После неё
  известна реальная скорость этой GPU. Если 20 эпох всех моделей этой GPU не успевают к `BUDGET_MIN` минутам,
  число эпох уменьшается, и косинус строится уже на новое число эпох. При скорости наших прогонов на Colab
  предохранитель не срабатывает: в логе будет «20/20».
* **Test только для предсказаний.** Снимки test читаются в разделе 3, но процессы обучения открывают их
  только после конца обучения, чтобы предсказать. Никаких статистик по test.
""")

# ------------------------------------------------------------------------------------------------ settings
md(r"""
## 1. Настройки
* `PLAN_2GPU` — какие модели обучает каждая GPU. Самая долгая модель идёт первой, тогда предохранитель
  времени сразу видит основную нагрузку.
* `PLAN_1GPU` — запасной план, если доступна только одна GPU: 3 модели по 16 эпох (около 31 минуты на T4).
* `BUDGET_MIN` — к этой минуте от старта таймера обучение должно закончиться (запас до 45 минут).
* `TTA_VIEWS` — сколько вариантов каждого тестового снимка усреднять: 8 = все повороты на 90° и отражения
  (так выбирали на val). Если преподаватели требуют предсказание test за 2 минуты (правило из исходного
  CPU-регламента), поставьте 2: предсказание займёт около 2 минут вместо 7, точность — чуть ниже (наша оценка).
* `QUICK_TEST = True` — проверка, что всё работает (1 эпоха, 260 снимков, 256 снимков test, 3–5 минут).
  Результат пишется в `final_submission_quick.csv`, а не в финальный файл.
""")
code(r"""
import time
NOTEBOOK_T0 = time.time()                     # для общего времени ноутбука
QUICK_TEST = False

PLAN_2GPU = {0: ["r152", "r50"], 1: ["r101", "inc3"]}
PLAN_1GPU = {0: ["r50", "inc3", "r101"]}
EPOCHS_2GPU, EPOCHS_1GPU = 20, 16
BUDGET_MIN = 40          # обучение должно закончиться к 40-й минуте от старта таймера (лимит 45)
TTA_VIEWS = 8            # 8 / 4 / 2 / 1
CUDNN_BENCHMARK = True   # False: детерминированные алгоритмы cuDNN (медленнее)
SEED = 42                # seed для Python, NumPy и torch задаётся в worker.py перед каждой моделью

# val accuracy holdout-моделей (обучены на train 20 эпох, проверены на val с TTA x8; Team_final.ipynb) — для таблицы
HOLDOUT_VAL = {"r50": 0.9742, "r101": 0.9700, "r152": 0.9714, "inc3": 0.9714, "inc3_363": 0.9771}
HOLDOUT_ENSEMBLE_VAL = {frozenset(["r50", "r101", "r152", "inc3"]): 0.9785,      # 15 ошибок из 699
                        frozenset(["r50", "r101", "inc3"]): 0.9728,              # 19 ошибок
                        frozenset(["r50", "inc3"]): 0.9757,                      # 17 ошибок
                        frozenset(["r50", "r152", "inc3", "inc3_363"]): 0.9814}  # 13 ошибок
""")

# ------------------------------------------------------------------------------------------------ environment
md(r"""
## 2. Окружение
Печатаем, на чём идёт обучение: число ядер CPU, память, видеокарты, версии библиотек. P100 не подходит:
torch в образе Kaggle собран без кода для этой видеокарты (ошибка «no kernel image is available»).
""")
code(r"""
import json
import os
import platform
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
from PIL import Image
import torch
import torchvision

n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "нет GPU: Settings -> Accelerator -> GPU T4 x2"
gpus = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
caps = [torch.cuda.get_device_capability(i) for i in range(n_gpu)]
if min(caps) < (7, 0):
    raise RuntimeError(f"{gpus}: эта версия torch не поддерживает P100 (sm_60). Выберите GPU T4 x2.")
ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
VERSIONS = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "pillow": PIL.__version__,
            "gpu": gpus, "cpu_count": os.cpu_count(), "ram_gb": round(ram_gb, 1)}
print(json.dumps(VERSIONS, ensure_ascii=False))
try:
    socket.create_connection(("download.pytorch.org", 443), timeout=5).close()
    print("интернет есть: веса ImageNet скачаются")
except OSError:
    print("!!! НЕТ ИНТЕРНЕТА: Settings -> Internet -> On (иначе веса ImageNet не скачать)")

PLAN = PLAN_2GPU if n_gpu >= 2 else PLAN_1GPU
EPOCHS = 1 if QUICK_TEST else (EPOCHS_2GPU if n_gpu >= 2 else EPOCHS_1GPU)
WORKERS_PER_GPU = max(1, os.cpu_count() // len(PLAN)) if len(PLAN) > 1 else max(2, os.cpu_count() - 1)
ALL_MODELS = [m for models_on_gpu in PLAN.values() for m in models_on_gpu]
print(f"план: {PLAN}, эпох: {EPOCHS}, процессов аугментации на GPU: {WORKERS_PER_GPU}")
""")

# ------------------------------------------------------------------------------------------------ worker
md(r"""
## 3. Код обучения одной GPU (`worker.py`) и подготовка test
Ячейка ниже записывает `worker.py` — тот же код, что в `Team_final.ipynb` (аугментации, модели, цикл
обучения, TTA), плюс предохранитель времени. Каждая GPU запускает его отдельным процессом со своим списком
моделей.
""")
code("%%writefile worker.py\n" + WORKER)

md(r"""
**Данные.** Ищем файлы соревнования в `/kaggle/input` (или в текущей папке, если запуск не на Kaggle).
Снимки **test** читаются здесь, до таймера обучения: это подготовка к предсказанию, а не обучение. Они
сохраняются в общий файл, который процессы обучения откроют только после конца обучения.
""")
code(r"""
def find_data_dir():
    found = [p.parent for p in Path("/kaggle/input").rglob("train.csv")] if Path("/kaggle/input").exists() else []
    found += [p for p in (Path("."), Path("data")) if (p / "train.csv").exists()]
    for p in found:
        if (p / "train").is_dir() and (p / "test").is_dir() and (p / "test.csv").exists():
            return p
    raise FileNotFoundError("не нашли train.csv / train / test: добавьте данные соревнования (Add Input)")

DATA_DIR = find_data_dir()
ARRAYS = Path("/tmp/lab5_arrays"); ARRAYS.mkdir(parents=True, exist_ok=True)   # большие массивы — не в Output
OUT = Path("runs45"); OUT.mkdir(exist_ok=True)

train_table = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
val_table = pd.read_csv(DATA_DIR / "val.csv", dtype={"id": str})
test_table = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})
classes = pd.read_csv(DATA_DIR / "classes.csv").sort_values("label").class_name.tolist()
if QUICK_TEST:   # по 10 train- и 3 val-снимка на класс, первые 256 снимков test
    train_table = train_table.groupby("label").head(10).reset_index(drop=True)
    val_table = val_table.groupby("label").head(3).reset_index(drop=True)
    test_table = test_table.head(256).reset_index(drop=True)

def read_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.BICUBIC)
        return np.asarray(img)

def load_images(table, folder):
    # снимки -> массив uint8 (N, 256, 256, 3); JPEG декодируются в несколько потоков
    paths = [DATA_DIR / folder / (i if i.endswith(".jpg") else f"{i}.jpg") for i in table.id]
    images = np.empty((len(paths), 256, 256, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for k, array in enumerate(pool.map(read_image, paths)):
            images[k] = array
    return images

start = time.time()
np.save(ARRAYS / "test_images.npy", load_images(test_table, "test"))
print(f"данные: {DATA_DIR} | train {len(train_table)}, val {len(val_table)}, test {len(test_table)}, "
      f"классов {len(classes)} | test прочитан за {(time.time() - start) / 60:.1f} мин")
""")

# ------------------------------------------------------------------------------------------------ training
md(r"""
## 4. Обучение (таймер)
**Что внутри таймера:** чтение train + val с диска, запуск процессов, скачивание весов ImageNet, все эпохи
всех моделей на обеих GPU. **Время обучения** = момент, когда закончила **последняя** GPU, минус старт
таймера. Это обычное время по часам (wall-clock): GPU работают одновременно, поэтому их минуты не
складываются.

**Что не входит:** чтение test (раздел 3) и предсказание test, его время печатается отдельно.

Раз в минуту печатается последняя строка лога каждой GPU; полные логи — `runs45/log_gpu*.txt`.
""")
code(r"""
T0 = time.time()                                   # ======== СТАРТ ТАЙМЕРА ОБУЧЕНИЯ ========
trainval_table = pd.concat([train_table, val_table], ignore_index=True)
np.save(ARRAYS / "trainval_images.npy", np.concatenate([load_images(train_table, "train"),
                                                         load_images(val_table, "val")]))
np.save(ARRAYS / "trainval_labels.npy", trainval_table.label.to_numpy())
print(f"train + val прочитаны: {len(trainval_table)} снимков, {(time.time() - T0) / 60:.1f} мин от старта")

procs, logs = {}, {}
for gpu, models_on_gpu in PLAN.items():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "worker.py", "--gpu", str(gpu), "--models", *models_on_gpu, "--t0", repr(T0),
           "--budget", str(BUDGET_MIN), "--epochs", str(EPOCHS), "--tta", str(TTA_VIEWS),
           "--workers", str(WORKERS_PER_GPU), "--benchmark", str(int(CUDNN_BENCHMARK)),
           "--arrays", str(ARRAYS), "--out", str(OUT)]
    logs[gpu] = OUT / f"log_gpu{gpu}.txt"
    procs[gpu] = subprocess.Popen(cmd, env=env, stdout=open(logs[gpu], "w"), stderr=subprocess.STDOUT)
    print(f"GPU {gpu}: запущено обучение {models_on_gpu}")

def last_line(path):
    lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    return lines[-1] if lines else "(пока пусто)"

def fail(gpu):
    print(f"!!! процесс GPU {gpu} упал, хвост лога:")
    print("\n".join(logs[gpu].read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
    for p in procs.values():
        p.kill()
    raise RuntimeError(f"обучение на GPU {gpu} упало")

train_end = {}
while len(train_end) < len(procs):
    time.sleep(20 if QUICK_TEST else 60)
    for gpu, p in procs.items():
        timing_file = OUT / f"timing_gpu{gpu}.json"
        if gpu not in train_end and timing_file.exists():
            train_end[gpu] = json.loads(timing_file.read_text(encoding="utf-8"))["t_train_end"]
        elif gpu not in train_end and p.poll() is not None:
            fail(gpu)
        print(last_line(logs[gpu]))

TRAIN_MIN = (max(train_end.values()) - T0) / 60   # ======== КОНЕЦ ТАЙМЕРА ОБУЧЕНИЯ ========
print("=" * 80)
print(f"ВРЕМЯ ОБУЧЕНИЯ ФИНАЛЬНОГО РЕШЕНИЯ: {TRAIN_MIN:.1f} мин (лимит 45 мин) | "
      + " | ".join(f"GPU {g}: {(t - T0) / 60:.1f} мин" for g, t in sorted(train_end.items())))
print("=" * 80)

for gpu, p in procs.items():                       # ждём предсказаний test
    if p.wait() != 0:
        fail(gpu)
timings = {gpu: json.loads((OUT / f"timing_gpu{gpu}.json").read_text(encoding="utf-8")) for gpu in procs}
PREDICT_MIN = (max(t["t_pred_end"] for t in timings.values()) - max(train_end.values())) / 60
print(f"предсказание test после конца обучения: {PREDICT_MIN:.1f} мин (TTA x{TTA_VIEWS})")
""")
md(r"""
**Как читать.** Строки `[guard]` в логах — прогноз времени после первой эпохи каждой модели. Если там «!!!»,
предохранитель уменьшил число эпох: решение уложилось в лимит, но качество может быть чуть ниже. Без «!!!»
все модели прошли полные 20 эпох, как в проверенном рецепте.
""")

# ------------------------------------------------------------------------------------------------ ensemble
md(r"""
## 5. Ансамбль и `final_submission.csv`
Вероятности четырёх моделей усредняются с равными весами. Веса не подбираем: на 699 снимках val это была бы
подгонка под шум. Метка — класс с наибольшей средней вероятностью.
""")
code(r"""
test_probs = {m: np.load(OUT / f"test_probs_{m}.npy") for m in ALL_MODELS}
ensemble = np.mean(list(test_probs.values()), axis=0)
final = pd.DataFrame({"id": test_table.id, "label": ensemble.argmax(1)})
assert list(final.columns) == ["id", "label"] and len(final) == len(test_table) == (256 if QUICK_TEST else 8299)
assert final.id.tolist() == test_table.id.tolist() and final.label.between(0, 19).all()
FINAL_NAME = "final_submission_quick.csv" if QUICK_TEST else "final_submission.csv"
final.to_csv(FINAL_NAME, index=False)
print(f"{FINAL_NAME}: {len(final)} строк; доли классов "
      f"{final.label.value_counts(normalize=True).min():.1%}...{final.label.value_counts(normalize=True).max():.1%}")
for m, p in test_probs.items():
    print(f"  {m:5s}: совпадает с ансамблем на {(p.argmax(1) == final.label.to_numpy()).mean():.2%} снимков test")
""")

# ------------------------------------------------------------------------------------------------ report
md(r"""
## 6. Итоговая таблица и время
`val acc` — точность holdout-модели с тем же рецептом (обучена на train, проверена на val, TTA ×8) из
`Team_final.ipynb`. Модели в этом ноутбуке обучены на train + val, проверить их на val уже нельзя.
""")
code(r"""
rows = []
for gpu, t in sorted(timings.items()):
    for m, info in t["per_model"].items():
        rows.append({"модель": m, "GPU": gpu, "эпох": info["epochs"], "обучение, мин": round(info["train_min"], 1),
                     "предсказание, мин": round(info["predict_min"], 1), "val acc (holdout)": HOLDOUT_VAL.get(m),
                     "seed": SEED})
rows.append({"модель": "ансамбль " + " + ".join(ALL_MODELS),
             "val acc (holdout)": HOLDOUT_ENSEMBLE_VAL.get(frozenset(ALL_MODELS))})
report = pd.DataFrame(rows)
report.to_csv("experiments_45min.csv", index=False)
TOTAL_MIN = (time.time() - NOTEBOOK_T0) / 60
summary = {"train_min_wall_clock": round(TRAIN_MIN, 2), "predict_min_after_training": round(PREDICT_MIN, 2),
           "notebook_total_min": round(TOTAL_MIN, 2), "plan": {str(k): v for k, v in PLAN.items()},
           "epochs": EPOCHS, "tta_views": TTA_VIEWS, "versions": VERSIONS}
Path("timing_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
print(report.to_string(index=False))
print(f"\nобучение {TRAIN_MIN:.1f} мин | предсказание {PREDICT_MIN:.1f} мин | весь ноутбук {TOTAL_MIN:.1f} мин")
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("cells:", len(cells), "->", OUT)
