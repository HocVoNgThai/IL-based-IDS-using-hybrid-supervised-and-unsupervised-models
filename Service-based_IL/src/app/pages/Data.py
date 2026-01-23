# pages/Data.py
import time
import gc
from pathlib import Path
from datetime import datetime
import shutil



import pandas as pd
import streamlit as st

from src.config.settings import settings
from src.config.incremental_config import incremental_settings
from src.app.config.func_convert import round_decimal
from src.app.config.config import (
    COLS_TO_DROP,
    MINMAX_COLS,
    STANDARD_COLS,
    DECIMAL_BIN,
)

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(layout="wide")

LABEL_OPTIONS = [
    "Benign", "Unknown", "DDoS", "DoS", "Reconnaisance",
    "MITM_ArpSpoofing", "DNS_Spoofing", "NeedManualLabel"
]

# =========================================================
# CACHE LOADERS
# =========================================================
@st.cache_data(show_spinner="📦 Loading dataset...")
def load_dataframe(path: str, mtime: float) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError("Unsupported file type")


@st.cache_data(show_spinner="📜 Loading JSONL alerts...")
def load_jsonl(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path.name


# =========================================================
# SESSION INIT
# =========================================================
def init_session():
    today = datetime.now().date()

    st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.13.1/font/bootstrap-icons.min.css">', unsafe_allow_html=True)
    st.session_state.setdefault(
        "current_file",
        str(settings.DATA_DIR / f"{today}" / "batch_0.parquet")
    )
    st.session_state.setdefault(
        "jsonl_file",
        str(settings.ALERTS_DIR / f"{today}.jsonl")
    )

    st.session_state.setdefault("df", None)
    st.session_state.setdefault("alerts_df_view", None)
    st.session_state.setdefault("overwrite_label", True)


# =========================================================
# MAIN
# =========================================================
def main():
    init_session()

    st.markdown('## <i class="bi bi-clipboard-data"></i> Data Management & Labeling', unsafe_allow_html=True)

    # =====================================================
    # SECTION A — DATASET
    # =====================================================
    st.markdown(
        f'### <i class="bi bi-graph-up-arrow"></i> Preprocessed Dataset | File: {Path(st.session_state.current_file).name}',
        unsafe_allow_html=True# Dir:{Path(settings.DATA_DIR).parent.resolve()}
    )

    colA1, colA2, colA3 = st.columns([4, 1, 1])

    with colA1:
        file_path = st.text_input(
            "| Dataset path (Parquet / CSV)", # "📂 Dataset path (Parquet / CSV)",
            value=st.session_state.current_file
        )

    with colA2:
        if st.button("Load File", type="primary", use_container_width=True):
            path = Path(file_path)
            if not path.exists():
                st.error("Không tìm thấy file!")
            else:
                st.session_state.current_file = str(path)
                try:
                    st.session_state.df = load_dataframe(
                        str(path), path.stat().st_mtime
                    )
                    st.success(f"Loaded: {path.name}")
                except Exception as e:
                    st.error(f"[ST] Error {e}")
                    print(e)
                
                time.sleep(0.2)
                st.rerun()

    # with colA3:
    #     save_mode = st.radio("Save Mode", ["Overwrite", "Save As"], horizontal=True)

    # =====================================================
    # SHOW + EDIT
    # =====================================================
    if st.session_state.df is not None:
        df = st.session_state.df

        df = round_decimal(
            df,
            min_max_cols=MINMAX_COLS,
            standard_cols=STANDARD_COLS,
            minmax_decimal_bin=8,
            standard_decimal_bin=DECIMAL_BIN
        )

        # st.subheader("📝 Edit Labels")
        st.markdown('### <i class="bi bi-pencil-square"></i> Edit Labels', unsafe_allow_html= True)
        df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "Label": st.column_config.SelectboxColumn(
                    "Label",
                    options=LABEL_OPTIONS,
                    required=True
                )
            }
        )

        st.session_state.df = df
        with st.container(border= True):
            # ---------------- Mapping ----------------
            with st.expander("Batch Label Mapping"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    from_label = st.selectbox("From", LABEL_OPTIONS)
                with c2:
                    to_label = st.selectbox("To", LABEL_OPTIONS)
                with c3:
                    if st.button("Apply Mapping"):
                        mask = df["Label"] == from_label
                        df.loc[mask, "Label"] = to_label
                        st.session_state.df = df
                        st.success(f"Mapped {mask.sum()} samples")
                        time.sleep(0.2)
                        st.rerun()

            # ---------------- Save ----------------
            st.markdown("### <i class='bi bi-floppy'></i> Save", unsafe_allow_html= True)

            c1, c2 = st.columns(2)

            with c1:
                if st.button(" Save (Overwrite)"):
                    fname = save_dataframe(
                        df, Path(st.session_state.current_file)
                    )
                    st.success(f"Saved → {fname}")

            with c2:
                curr_file_time = datetime.fromtimestamp(Path(st.session_state.current_file).stat().st_mtime)
                curr_file_time = curr_file_time.strftime("%Y-%m-%d %H-%M-%S")
                save_as = st.text_input(
                    "Save As _ to train_data _ For IL",
                    value= incremental_settings.IL_DATA_DIR/ f"{curr_file_time}.parquet"
                )
                if st.button("Save As"):
                    out = Path (settings.DATA_DIR / save_as)
                    fname = save_dataframe(df, out)
                    st.success(f"Saved As→ {out}")
    
    # with st.container(border = True):
    #     st.markdown(f"### <i class='bi bi-copy'></i> Copy file từ {str(settings.DATA_DIR / str(datetime.now().date()))}/ sang {incremental_settings.IL_DATA_DIR}/",
    #                 unsafe_allow_html=True)
    #     selected_files = st.multiselect("Chọn các tập tin để copy số lượng lớn:", list(Path(settings.DATA_DIR / str(datetime.now().date())).glob("*.parquet")))
        
    #     if st.button("Execute"):
    #         for f in selected_files:
    #             if not f.exists():
    #                 continue
    #             ftime = datetime.fromtimestamp(f.stat().st_mtime)
    #             ftime = ftime.strftime("%Y-%m-%d %H-%M-%S")
                
    #             shutil.copy(f.resolve(), incremental_settings.IL_DATA_DIR/ f"{ftime}.parquet")
            
    #         st.success("Copy Success!")
    
    with st.container(border=True):
        st.markdown(f"### <i class='bi bi-copy'></i> Move file dữ liệu hàng loạt", unsafe_allow_html=True)
        
        # 1. Lấy danh sách các thư mục con trong DATA_DIR (thường là các thư mục ngày tháng)
        base_path = Path(settings.DATA_DIR)
        if base_path.exists():
            sub_dirs = [d.name for d in base_path.iterdir() if d.is_dir()]
            sub_dirs.sort(reverse=True) # Sắp xếp mới nhất lên đầu
        else:
            sub_dirs = []

        # 2. Cho phép người dùng chọn thư mục nguồn
        selected_dir_name = st.selectbox("Chọn thư mục nguồn (Ngày):", sub_dirs)
        
        if selected_dir_name:
            source_path = base_path / selected_dir_name
            st.info(f"Đang quét file tại: `{source_path}`")
            
            # 3. Lấy danh sách file từ thư mục đã chọn
            files_in_dir = list(source_path.glob("*.parquet"))
            
            # Hiển thị tên file cho gọn trong multiselect
            file_options = {f.name: f for f in files_in_dir}
            
            selected_file_names = st.multiselect(
                "Chọn các tập tin để Move:", 
                options=list(file_options.keys())
            )

            if st.button("Execute Move"):
                if not selected_file_names:
                    st.warning("Vui lòng chọn ít nhất một file.")
                else:
                    progress_bar = st.progress(0)
                    for idx, name in enumerate(selected_file_names):
                        f = file_options[name]
                        if not f.exists():
                            continue

                        target_name = f"{f.name}"
                        shutil.move(f.resolve(), Path(incremental_settings.IL_DATA_DIR) / target_name)
                        
                        # Cập nhật progress bar
                        progress_bar.progress((idx + 1) / len(selected_file_names))
                    
                    st.success(f"Đã move thành công {len(selected_file_names)} file vào thư mục Incremental {incremental_settings.IL_DATA_DIR}!")
                    
    # =====================================================
    # SECTION B — JSONL ALERTS
    # =====================================================
    st.markdown("---")
    st.markdown(
        f"### <i class='bi bi-list-ul'></i> Raw Flow Alerts | File: {Path(st.session_state.jsonl_file).name}",
        unsafe_allow_html=True
    )

    colB1, colB2, colB3 = st.columns([4, 1, 1])

    with colB1:
        json_path = st.text_input(
            "| JSONL path",
            value=st.session_state.jsonl_file
        )

    with colB2:
        if st.button("Load Alerts", type="primary", use_container_width=True):
            path = Path(json_path)
            if not path.exists():
                st.error("Không tìm thấy JSONL file!")
            else:
                st.session_state.jsonl_file = str(path)
                alerts_df = load_jsonl(
                    str(path), path.stat().st_mtime
                )

                flow_ids = set(st.session_state.df["Flow ID"])
                st.session_state.alerts_df_view = alerts_df[
                    alerts_df["Flow ID"].isin(flow_ids)
                ]

                del alerts_df, flow_ids
                gc.collect()

                st.success("Alerts loaded")
                time.sleep(0.2)
                st.rerun()

    with colB3:
        overwrite = st.checkbox(
            "Overwrite existing labels",
            value=st.session_state.overwrite_label
        )
        st.session_state.overwrite_label = overwrite

        if st.button("Apply Mapping", type="primary", use_container_width=True):
            df = st.session_state.df
            alerts = st.session_state.alerts_df_view

            if alerts is None:
                st.error("No alerts loaded!")
            else:
                mapping = (
                    alerts
                    .dropna(subset=["Flow ID", "Label"])
                    .drop_duplicates("Flow ID")
                    .set_index("Flow ID")["Label"]
                    .to_dict()
                )

                if overwrite:
                    mask = df["Flow ID"].isin(mapping)
                else:
                    mask = (
                        df["Flow ID"].isin(mapping)
                        & df["Label"].isin(["NeedManualLabel", "Unknown"])
                    )

                count = mask.sum()
                df.loc[mask, "Label"] = df.loc[mask, "Flow ID"].map(mapping)
                st.session_state.df = df

                st.success(f"✔ Mapped {count} flows")
                time.sleep(0.5)
                st.rerun()

    if st.session_state.alerts_df_view is not None:
        st.caption(
            f"Matched Flow IDs: {len(st.session_state.alerts_df_view)}"
        )
        st.dataframe(
            st.session_state.alerts_df_view,
            use_container_width=True,
            hide_index=True
        )
# # =========================================================
# if __name__ == "__main__":
#     main()
