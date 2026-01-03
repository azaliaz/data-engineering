import os
import subprocess
import shlex
from pathlib import Path
import re
import mimetypes
import base64
import shutil
import uuid

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
DATA_PROCESSED_STOCKS = SRC / "data" / "processed" / "stocks"
DATA_PROCESSED_CRYPTO = SRC / "data" / "processed" / "crypto"

SRC_REPORTS_ROOT = SRC / "reports"
PROJECT_REPORTS_ROOT = ROOT / "reports"

REPORTS_FIGS_PRIMARY = SRC_REPORTS_ROOT / "figures"
REPORTS_TABLES_PRIMARY = SRC_REPORTS_ROOT / "tables"

REPORTS_TABLES_DEST = PROJECT_REPORTS_ROOT / "tables"
REPORTS_TABLES_DEST.mkdir(parents=True, exist_ok=True)

REPORTS_FIGS_ALIASES = [
    REPORTS_FIGS_PRIMARY,
    SRC_REPORTS_ROOT / "figures_cor",
    SRC_REPORTS_ROOT / "figures_top10",
    SRC_REPORTS_ROOT / "figures_plot_single",
    PROJECT_REPORTS_ROOT / "figures",
    PROJECT_REPORTS_ROOT / "figures_top10",
]
REPORTS_TABLES_ALIASES = [
    REPORTS_TABLES_PRIMARY,
    SRC_REPORTS_ROOT / "tables_top10",
    PROJECT_REPORTS_ROOT / "tables",
    PROJECT_REPORTS_ROOT / "tables_top10",
]

for d in [
    REPORTS_FIGS_PRIMARY,
    REPORTS_TABLES_PRIMARY,
    SRC_REPORTS_ROOT / "figures_top10",
    SRC_REPORTS_ROOT / "tables_top10",
    PROJECT_REPORTS_ROOT / "tables",
    PROJECT_REPORTS_ROOT / "figures_top10",
]:
    d.mkdir(parents=True, exist_ok=True)


def make_image_data_url(path: Path) -> str:
    try:
        ext = path.suffix.lower().lstrip('.')
        mime = "image/png" if ext == "png" else f"image/{ext}"
        with open(path, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


def run_cmd(cmd, cwd=ROOT, env=None, timeout=None):
    """Запустить команду и вернуть (returncode, stdout+stderr)."""
    try:
        if isinstance(cmd, (list, tuple)):
            proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env, timeout=timeout)
        else:
            proc = subprocess.run(shlex.split(cmd), cwd=str(cwd), capture_output=True, text=True, env=env, timeout=timeout)
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out
    except subprocess.TimeoutExpired as e:
        return 124, f"TimeoutExpired: {e}"
    except Exception as e:
        return 1, f"Exception running command: {e}"


def list_csv(folders):
    """Возвращает список CSV-файлов (не рекурсивно)."""
    if not isinstance(folders, (list, tuple)):
        folders = [folders]
    out = []
    for folder in folders:
        if folder and Path(folder).exists():
            out.extend(sorted([str(x) for x in Path(folder).glob("*.csv")]))
    return out


def list_images(folders):
    """Рекурсивно ищет изображения в папках и подпапках алиасов; учитывает регистр расширений."""
    if not isinstance(folders, (list, tuple)):
        folders = [folders]
    imgs = []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    for folder in folders:
        if folder and Path(folder).exists():
            for ext in exts:
                imgs.extend([str(x.resolve()) for x in Path(folder).rglob(ext)])
    seen = set()
    unique = []
    for p in imgs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    unique.sort()
    return unique


def find_assets():
    stocks = list_csv([DATA_PROCESSED_STOCKS])
    crypto = list_csv([DATA_PROCESSED_CRYPTO])
    report_tables_csvs = list_csv(REPORTS_TABLES_ALIASES)
    return {"stocks": stocks, "crypto": crypto, "report_tables": report_tables_csvs}


def parse_saved_paths(text):
    if not text:
        return []
    pattern = re.compile(r"Saved\s+(?:->\s*)?([\S]+?\.(?:csv|png|jpe?g|md))", flags=re.IGNORECASE)
    matches = pattern.findall(text)
    pattern2 = re.compile(r"->\s*([\S]+?\.(?:csv|png|jpe?g|md))", flags=re.IGNORECASE)
    matches += pattern2.findall(text)
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def resolve_paths(matches):
    candidates = []
    search_dirs = [ROOT, SRC, REPORTS_TABLES_PRIMARY, REPORTS_FIGS_PRIMARY]
    search_dirs += REPORTS_FIGS_ALIASES + REPORTS_TABLES_ALIASES
    for m in matches:
        p = Path(m)
        found = None
        if p.exists():
            found = p.resolve()
        else:
            for d in search_dirs:
                try:
                    trial = (ROOT / p) if d is None else (Path(d) / p.name)
                    if trial.exists():
                        found = trial.resolve()
                        break
                except Exception:
                    continue
            if not found:
                for root_try in [PROJECT_REPORTS_ROOT, SRC_REPORTS_ROOT, SRC]:
                    try:
                        hits = list(root_try.rglob(p.name)) if root_try.exists() else []
                        if hits:
                            found = hits[0].resolve()
                            break
                    except Exception:
                        continue
        if found:
            candidates.append(found)
    return candidates


def _sanitize_key(s: str) -> str:
    return re.sub(r'[^0-9a-zA-Z_\-]', '_', s)


def display_paths(paths):
    for p_raw in paths:
        p = Path(p_raw)
        suffix = p.suffix.lower()
        st.markdown(f"**Найдено и отображается:** `{p}`")
        try:
            if suffix == ".csv":
                try:
                    df = pd.read_csv(p)
                    st.dataframe(df)
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    unique = uuid.uuid4().hex
                    key = f"download_csv_{_sanitize_key(str(p))}_{unique}"
                    st.download_button(label=f"Скачать {p.name}", data=csv_bytes, file_name=p.name, mime='text/csv', key=key)
                except Exception as e:
                    st.error(f"Не удалось открыть CSV {p}: {e}")
            elif suffix in (".png", ".jpg", ".jpeg"):
                st.image(str(p), caption=p.name, use_container_width=True)
                data_url = make_image_data_url(p)
                if data_url:
                    st.markdown(f'<a href="{data_url}" target="_blank" rel="noopener noreferrer">Открыть в новой вкладке (полный размер)</a>', unsafe_allow_html=True)
                try:
                    with open(p, "rb") as f:
                        data = f.read()
                    unique = uuid.uuid4().hex
                    key = f"download_img_{_sanitize_key(str(p))}_{unique}"
                    st.download_button(label=f"Скачать {p.name}", data=data, file_name=p.name, key=key)
                except Exception as e:
                    st.error(f"Ошибка скачивания изображения {p}: {e}")
            elif suffix == ".md":
                try:
                    text = p.read_text(encoding="utf-8")
                    st.markdown(text, unsafe_allow_html=True)
                    unique = uuid.uuid4().hex
                    key = f"download_md_{_sanitize_key(str(p))}_{unique}"
                    st.download_button(label=f"Скачать {p.name}", data=text.encode('utf-8'), file_name=p.name, mime='text/markdown', key=key)
                except Exception as e:
                    st.error(f"Не удалось прочитать markdown {p}: {e}")
            else:
                st.write(f"Файл найден: {p} (тип {suffix}), откройте локально.")
        except Exception as e:
            st.error(f"Ошибка отображения {p}: {e}")


def plot_asset_matplotlib(df, asset_name, fig_dir=REPORTS_FIGS_PRIMARY):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    fig1, ax1 = plt.subplots(figsize=(9, 3.5))
    price_col = df['price'] if 'price' in df.columns else df.get('Adj Close')
    ax1.plot(df.index, price_col, label='Price', linewidth=2)
    ax1.set_title(f"{asset_name} Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for lbl in ax1.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha('right')
    fig1.tight_layout()
    fig1.autofmt_xdate()

    vcol = 'volatility_20' if 'volatility_20' in df.columns else next((c for c in df.columns if c.startswith("volatility")), None)
    fig2 = None
    if vcol:
        fig2, ax2 = plt.subplots(figsize=(9, 3.5))
        ax2.plot(df.index, df[vcol], label=vcol, linewidth=2)
        ax2.set_title(f"{asset_name} Volatility")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volatility")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        for lbl in ax2.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha('right')
        fig2.tight_layout()
        fig2.autofmt_xdate()

    safe_name = asset_name.replace("/", "_")
    fig1.savefig(Path(fig_dir) / f"{safe_name}_price.png", bbox_inches='tight', dpi=150)
    plt.close(fig1)
    if fig2:
        fig2.savefig(Path(fig_dir) / f"{safe_name}_vol.png", bbox_inches='tight', dpi=150)
        plt.close(fig2)
    return fig1, fig2

def copy_tables_to_project_reports(source_dirs=None, dest=REPORTS_TABLES_DEST):
    if source_dirs is None:
        source_dirs = [
            SRC_REPORTS_ROOT / "tables",
            SRC_REPORTS_ROOT / "tables_top10",
        ]

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []

    for sd in source_dirs:
        sd = Path(sd)
        if not sd.exists():
            continue
        for f in sd.glob("*.csv"):
            try:
                target = dest / f.name
                if f.resolve() == target.resolve():
                    continue

                if target.exists():
                    try:
                        if f.stat().st_size == target.stat().st_size and int(f.stat().st_mtime) == int(target.stat().st_mtime):
                            continue
                    except Exception:
                        pass

                shutil.copy2(str(f), str(target))
                copied.append(target)
            except Exception:
                continue

    return copied




st.set_page_config(page_title="Analytics Platform", layout="wide")
st.title("Аналитическая платформа")

st.sidebar.header("Действия")
action = st.sidebar.selectbox("Выберите действие", (
    "—",
    "Preprocess",
    "Load to DB",
    "Run analysis & metrics",
    "Generate plots (price-vol)",
    "Generate top10",
    "Correlation heatmaps",
    "Data Quality report"
))
st.sidebar.markdown("---")
st.sidebar.markdown("### Файлы и отчёты")
if st.sidebar.button("Открыть папку reports/figures (все алиасы)"):
    imgs = list_images(REPORTS_FIGS_ALIASES)
    st.write(imgs[:200])
st.sidebar.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    if action == "Run analysis & metrics":
        st.write("Запуск аналитики: src/scripts/summary_metrics.py")
        if st.button("Run summary_metrics.py"):
            with st.spinner("Запуск summary_metrics..."):
                code, out = run_cmd(["python3", str(SRC / "scripts" / "summary_metrics.py")])
                st.text_area("Лог summary", out, height=400)

                matches = parse_saved_paths(out)
                paths = resolve_paths(matches)
                if paths:
                    st.markdown("### Файлы, упомянутые в логе:")
                    display_paths(paths)

                copied = copy_tables_to_project_reports()
                if copied:
                    st.markdown("### Скопированные таблицы в проектный reports/tables:")
                    display_paths(copied)
                    st.success(f"Скопировано {len(copied)} таблиц в {REPORTS_TABLES_DEST}")
                else:
                    st.info("Не найдено таблиц для копирования в reports/tables.")

                if code == 0:
                    st.success("Аналитика выполнена")
                else:
                    st.error("Ошибка аналитики")

    elif action == "Data Quality report":
        st.write("Запустить data_quality.py (src/data_quality.py)")
        if st.button("Run data_quality.py"):
            with st.spinner("Проверка качества данных..."):
                code, out = run_cmd(["python3", str(SRC / "data_quality.py")])
                st.text_area("Лог DQ", out, height=400)

                matches = parse_saved_paths(out)
                paths = resolve_paths(matches)
                if paths:
                    st.markdown("### Файлы, упомянутые в логе:")
                    display_paths(paths)

                copied = copy_tables_to_project_reports()
                if copied:
                    st.markdown("### Скопированные таблицы в проектный reports/tables:")
                    display_paths(copied)
                    st.success(f"Скопировано {len(copied)} таблиц в {REPORTS_TABLES_DEST}")
                else:
                    st.info("Не найдено таблиц для копирования в reports/tables.")

                if code == 0:
                    st.success("Data Quality выполнен")
                else:
                    st.warning("DQ script завершился с предупреждениями/ошибками")

    elif action == "Preprocess":
        st.write("Предобработка: src/preprocessing.py")
        if st.button("Start preprocessing.py"):
            with st.spinner("Предобработка..."):
                code, out = run_cmd(["python3", str(SRC / "preprocessing.py")])
                st.text_area("Лог предобработки", out, height=400)
                matches = parse_saved_paths(out)
                paths = resolve_paths(matches)
                if paths:
                    st.markdown("### Авто-отображение файлов (из лога)")
                    display_paths(paths)
                if code == 0:
                    st.success("Предобработка завершена")
                else:
                    st.error(f"Ошибка предобработки (код {code})")

    elif action == "Load to DB":
        st.write("Загрузить обработанные данные в БД: src/load_to_db.py")
        st.write("DATABASE_URL: ", os.getenv("DATABASE_URL", "(не установлено)"))
        if st.button("Start load_to_db.py"):
            with st.spinner("Загрузка в БД..."):
                code, out = run_cmd(["python3", str(SRC / "load_to_db.py")])
                st.text_area("Лог загрузки в БД", out, height=400)
                matches = parse_saved_paths(out)
                paths = resolve_paths(matches)
                if paths:
                    display_paths(paths)
                if code == 0:
                    st.success("Загрузка в БД завершена")
                else:
                    st.error(f"Ошибка загрузки (код {code})")

    elif action == "Generate top10":
        st.write("Запуск src/scripts/top_return.py (Top10 return/vol и фигуры)")
        if st.button("Run top_return.py"):
            with st.spinner("Генерируем top10 tables & figures..."):
                code, out = run_cmd(["python3", str(SRC / "scripts" / "top_return.py")])
                st.text_area("Лог top_return", out, height=400)
                figs = list_images([SRC_REPORTS_ROOT / "figures_top10", PROJECT_REPORTS_ROOT / "figures_top10"])
                tables = list_csv([SRC_REPORTS_ROOT / "tables_top10", PROJECT_REPORTS_ROOT / "tables_top10"])
                if figs:
                    st.markdown("### Сгенерированные фигуры top10:")
                    for f in figs:
                        p = Path(f)
                        st.image(str(p), use_container_width=True)
                        url = make_image_data_url(p)
                        if url:
                            st.markdown(f'<a href="{url}" target="_blank" rel="noopener noreferrer">Открыть в новой вкладке (полный размер)</a>', unsafe_allow_html=True)
                else:
                    st.info("Фигур top10 не найдено.")
                if tables:
                    st.markdown("### Сгенерированные таблицы top10:")
                    for t in tables:
                        try:
                            df = pd.read_csv(t)
                            st.write(Path(t).name)
                            st.dataframe(df)
                            csv_bytes = df.to_csv(index=False).encode('utf-8')
                            key = f"download_top10_{_sanitize_key(str(t))}_{uuid.uuid4().hex}"
                            st.download_button(label=f"Скачать {Path(t).name}", data=csv_bytes, file_name=Path(t).name, mime='text/csv', key=key)
                        except Exception as e:
                            st.error(f"Не удалось открыть {t}: {e}")
                if code == 0:
                    st.success("top_return.py выполнен")
                else:
                    st.error("top_return.py завершился с ошибкой")

    elif action == "Generate plots (price-vol)":
        st.write("Генерация графиков: src/plots_pricevol.py")
        if st.button("Generate & price-vol plots"):
            with st.spinner("Генерация графиков..."):
                code, out = run_cmd(["python3", str(SRC / "plots_pricevol.py")])
                st.text_area("Лог генерации", out, height=400)
                matches = parse_saved_paths(out)
                paths = resolve_paths(matches)
                if paths:
                    display_paths(paths)
                if code == 0:
                    st.success("Графики сохранены")
                else:
                    st.error("Ошибка генерации графиков")

    elif action == "Correlation heatmaps":
        st.write("Построить корреляционные теплокарты: src/plots_corr_heatmaps.py")
        if st.button("Build correlation heatmaps"):
            with st.spinner("Строим корреляции..."):
                code, out = run_cmd(["python3", str(SRC / "plots_corr_heatmaps.py")])
                st.text_area("Лог корреляций", out, height=400)
                matches = parse_saved_paths(out)
                paths = resolve_paths(matches)
                if paths:
                    display_paths(paths)
                if code == 0:
                    st.success("Корреляции построены")
                else:
                    st.error("Ошибка при построении корреляций")

with col2:
    st.subheader("Просмотр отчётов и графиков")
    st.markdown("### Выберите тип актива и тикер (акции / крипто)")
    assets = find_assets()
    kind = st.radio("Категория", options=["stocks", "crypto", "both"], index=0, horizontal=True)

    if kind == "stocks":
        available = assets["stocks"]
    elif kind == "crypto":
        available = assets["crypto"]
    else:
        available = assets["stocks"] + assets["crypto"]

    if not available:
        st.info("Нет CSV файлов в data/processed/stocks или data/processed/crypto. Сгенерируйте данные сначала.")
    else:
        labels = [Path(p).stem.replace("_usd", "") + (" — crypto" if "crypto" in p else " — stock") for p in available]
        sel_index = st.selectbox("Доступные активы", options=list(range(len(available))), format_func=lambda i: labels[i])
        sel_path = Path(available[sel_index])

        st.markdown("#### Просмотр таблицы выбранного CSV")
        try:
            df = pd.read_csv(sel_path, index_col=0, parse_dates=True)
            st.dataframe(df.head(200))
            if st.button("Построить графики для выбранного актива"):
                with st.spinner("Строим графики..."):
                    try:
                        asset_name = sel_path.stem.replace("_usd", "")
                        fig_price, fig_vol = plot_asset_matplotlib(df, asset_name)
                        st.success("Графики построены и сохранены в reports/figures/ (primary)")

                        price_path = SRC_REPORTS_ROOT / "figures" / f"{asset_name}_price.png"
                        if price_path.exists():
                            st.image(str(price_path), use_container_width=True)
                            url = make_image_data_url(price_path)
                            if url:
                                st.markdown(f'<a href="{url}" target="_blank" rel="noopener noreferrer">Открыть price в новой вкладке (полный размер)</a>', unsafe_allow_html=True)

                        vol_path = SRC_REPORTS_ROOT / "figures" / f"{asset_name}_vol.png"
                        if vol_path.exists():
                            st.image(str(vol_path), use_container_width=True)
                            url2 = make_image_data_url(vol_path)
                            if url2:
                                st.markdown(f'<a href="{url2}" target="_blank" rel="noopener noreferrer">Открыть vol в новой вкладке (полный размер)</a>', unsafe_allow_html=True)
                        else:
                            st.info("Volatility chart not available for this asset.")
                    except Exception as e:
                        st.error(f"Ошибка при построении графиков: {e}")
        except Exception as e:
            st.error(f"Не удалось прочитать CSV {sel_path}: {e}")

    st.markdown("---")
    st.markdown("### Графики, сохранённые в reports/figures (включая алиасы)")
    imgs = list_images(REPORTS_FIGS_ALIASES)
    if imgs:
        img_options = []
        for i in imgs:
            p = Path(i)
            rel = None
            try:
                rel = p.relative_to(SRC_REPORTS_ROOT)
            except Exception:
                try:
                    rel = p.relative_to(PROJECT_REPORTS_ROOT)
                except Exception:
                    rel = Path(p.name)
            img_options.append(str(rel))
        sel_rel = st.selectbox("Выберите файл изображения", img_options)
        if sel_rel:
            candidate1 = SRC_REPORTS_ROOT / sel_rel
            candidate2 = PROJECT_REPORTS_ROOT / sel_rel
            img_path = candidate1 if candidate1.exists() else (candidate2 if candidate2.exists() else None)
            if img_path and img_path.exists():
                st.image(str(img_path), use_container_width=True)
                data_url = make_image_data_url(img_path)
                if data_url:
                    st.markdown(f'<a href="{data_url}" target="_blank" rel="noopener noreferrer">Открыть в новой вкладке (полный размер)</a>', unsafe_allow_html=True)
                with open(img_path, 'rb') as f:
                    key = f"download_img_list_{_sanitize_key(str(img_path))}_{uuid.uuid4().hex}"
                    st.download_button(label="Скачать изображение", data=f.read(), file_name=Path(sel_rel).name, key=key)
            else:
                st.error("Не удалось найти выбранное изображение в известных папках.")
    else:
        st.info("Папки reports/figures* пусты.")

st.markdown("---")
st.caption("Интерфейс запускает локальные скрипты и показывает сгенерированные файлы.")
