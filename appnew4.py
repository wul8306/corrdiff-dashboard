import json
import os
import sqlite3
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

# ==========================================
# 0. 設定 Matplotlib 中文字型 (無需 sudo 權限)
# ==========================================
FONT_PATH = "font.ttf"

if os.path.exists(FONT_PATH):
  my_font = fm.FontProperties(fname=FONT_PATH)
  fm.fontManager.addfont(FONT_PATH)
  plt.rcParams["font.sans-serif"] = [my_font.get_name()]
else:
  plt.rcParams["font.sans-serif"] = [
      "Noto Sans CJK TC",
      "Microsoft JhengHei",
      "DejaVu Sans",
  ]

plt.rcParams["axes.unicode_minus"] = False  # 避免負號顯示異常


# ==========================================
# 1. 頁面基本配置
# ==========================================
st.set_page_config(
    page_title="CorrDiff 降尺度實驗評估系統", page_icon="🌧️", layout="wide"
)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "corrdiff_allmetrics.db")
BASE_NC_DIR = "output"
EXPERIMENT_NC_MAP = {
    "P2P": os.path.join(BASE_NC_DIR, "P2P25Y/output_0_all.nc"),
    "CP2P": os.path.join(BASE_NC_DIR, "CP2P25Y/output_0_all.nc"),
}

# WRF 208x208 網格與地形座標檔
GRID_COORDS_NC = "wrf_208x208_grid_coords.nc"


# ==========================================
# 2. CWB 標準雨量色階與 BoundaryNorm 定義
# ==========================================
PRECIP_BOUNDS = [0, 5, 10, 20, 30, 50, 60, 90, 130, 150, 200, 270, 300, 400, 500, 600, 700]
PRECIP_COLORS = [
    "#FFFFFF", "#A0F0FF", "#00A0FF", "#0060FF", "#0000FF", "#00A000", "#00E000", "#FFFF00",
    "#FFC000", "#FF8000", "#FF0000", "#D00000", "#960000", "#6E006E", "#B400D2", "#FF00FF", "#FFC0FF"
]
cwb_cmap = LinearSegmentedColormap.from_list(
    "cwb_precip_cmap", PRECIP_COLORS, N=len(PRECIP_COLORS)
)
cwb_norm = BoundaryNorm(PRECIP_BOUNDS, ncolors=cwb_cmap.N, extend="max")


# ==========================================
# 3. 快取載入與資料庫讀取函式
# ==========================================
@st.cache_data
def load_wrf_grid_coords(coords_path=GRID_COORDS_NC):
  """載入專屬 WRF 網格座標 (XLAT, XLONG, LANDMASK, TER)"""
  if not os.path.exists(coords_path):
    return None, None, None, None
  with xr.open_dataset(coords_path) as ds_coords:
    lats = ds_coords["XLAT"].values
    lons = ds_coords["XLONG"].values
    land_mask = ds_coords["LANDMASK"].values
    ter = ds_coords["TER"].values
  return lats, lons, land_mask, ter


@st.cache_data
def load_all_time_metrics(db_path=DB_PATH):
  """載入所有實驗的全時間平均指標"""
  print(db_path)
  if not os.path.exists(db_path):
    return None
  conn = sqlite3.connect(db_path)
  query = """
    SELECT exp_name, file_name, comparison_type, variable_name,
           rmse, fss_threshold, fss_window_size, fss_score
    FROM summary_metrics
    WHERE comparison_type = 'pred_vs_truth'
    """
  try:
    df = pd.read_sql_query(query, conn)
  except Exception:
    df = pd.DataFrame()
  finally:
    conn.close()
  return df


@st.cache_data
def get_scatter_combos(table_name="grid_rmse_scatter", db_path=DB_PATH):
  """動態讀取指定 scatter 表格 (grid_rmse_scatter 或 time_rmse_scatter) 中的實驗組合"""
  if not os.path.exists(db_path):
    return []
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  try:
    cursor.execute(f"SELECT DISTINCT exp_x, exp_y FROM {table_name}")
    rows = cursor.fetchall()
    return rows
  except Exception:
    return []
  finally:
    conn.close()


def load_scatter_data(
    table_name, exp_x, exp_y, var_name="precipitation", db_path=DB_PATH
):
  """從指定的 scatter 資料表讀取兩個實驗的散佈數據"""
  if not os.path.exists(db_path):
    return None, None, None, None
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  try:
    cursor.execute(
        f"SELECT valid_time, ratio_y_gt_x, x_values, y_values FROM {table_name}"
        " WHERE exp_x=? AND exp_y=? AND var_name=?",
        (exp_x, exp_y, var_name),
    )
    row = cursor.fetchone()

    is_swapped = False
    if not row:
      cursor.execute(
          f"SELECT valid_time, ratio_y_gt_x, x_values, y_values FROM"
          f" {table_name} WHERE exp_x=? AND exp_y=? AND var_name=?",
          (exp_y, exp_x, var_name),
      )
      row = cursor.fetchone()
      is_swapped = True
  except Exception:
    row = None
  finally:
    conn.close()

  if not row:
    return None, None, None, None

  raw_valid_time = row[0]
  try:
      # 只有時間軸層級 (time_rmse_scatter) 的 JSON 字串才能成功轉為 List
      valid_time = json.loads(raw_valid_time)
      valid_time = tuple(valid_time)
  except (json.JSONDecodeError, TypeError):
      # 格點層級 (grid_rmse_scatter) 為一般字串 'GRID_POINTS'，直接保留原值
      valid_time = raw_valid_time

  ratio_orig = row[1]
  x_draw = np.frombuffer(row[2], dtype=np.float32)
  y_draw = np.frombuffer(row[3], dtype=np.float32)

  if is_swapped:
    x_draw, y_draw = y_draw, x_draw
    ratio_y_gt_x = 100.0 - ratio_orig
  else:
    ratio_y_gt_x = ratio_orig

  return valid_time, ratio_y_gt_x, x_draw, y_draw


@st.cache_data
def get_nc_time_list(nc_path):
  """解析相對時間天數並轉為 YYYY-MM-DD HH:MM (用於 TAB 3)"""
  if not os.path.exists(nc_path):
    return []
  with xr.open_dataset(nc_path) as ds:
    try:
      if np.issubdtype(ds["time"].dtype, np.datetime64):
        time_series = pd.to_datetime(ds["time"].values)
      else:
        base_date = pd.Timestamp("2016-01-01 00:00:00")
        time_series = [
            base_date + pd.Timedelta(days=float(t)) for t in ds["time"].values
        ]
      times = [t.strftime("%Y-%m-%d %H:%M") for t in time_series]
    except Exception:
      times = [f"Step {idx}" for idx in range(len(ds["time"].values))]
  return times


@st.cache_data
def load_spatial_slice(nc_path, time_idx, var_name):
  """讀取 Truth, Prediction (Ensemble平均), Input 的 2D 矩陣"""
  with (
      xr.open_dataset(nc_path, group="truth") as ds_truth,
      xr.open_dataset(nc_path, group="prediction") as ds_pred,
      xr.open_dataset(nc_path, group="input") as ds_input,
  ):

    truth_2d = ds_truth[var_name].isel(time=time_idx).values
    input_2d = ds_input[var_name].isel(time=time_idx).values

    pred_var = ds_pred[var_name].isel(time=time_idx)
    if "ensemble" in pred_var.dims:
      pred_2d = pred_var.mean(dim="ensemble").values
    else:
      pred_2d = pred_var.values

  return truth_2d, pred_2d, input_2d


# ==========================================
# 4. 通用散佈圖渲染模組 (供 TAB 2 呼叫)
# ==========================================
def render_scatter_plot(table_name, mode_name):
  combos = get_scatter_combos(table_name, DB_PATH)
  if not combos:
    st.error(f"❌ 資料庫 `{DB_PATH}` 中找不到 `{table_name}` 的預算資料。")
    return

  all_exps = sorted(list(set([r[0] for r in combos] + [r[1] for r in combos])))

  col_x, col_y = st.columns(2)
  with col_x:
    exp_x = st.selectbox(
        f"選擇 X 軸實驗 (Baseline)",
        all_exps,
        index=0,
        key=f"{table_name}_x",
    )
  with col_y:
    exp_y = st.selectbox(
        f"選擇 Y 軸實驗 (Comparison)",
        all_exps,
        index=1 if len(all_exps) > 1 else 0,
        key=f"{table_name}_y",
    )

  if exp_x == exp_y:
    st.warning("⚠️ 請選擇不同的實驗進行比較。")
    return

  valid_time, ratio_y_gt_x, x_draw, y_draw = load_scatter_data(
      table_name, exp_x, exp_y, var_name="precipitation", db_path=DB_PATH
  )

  if x_draw is not None and len(x_draw) > 0:
#   date_info = [valid_time if valid_time else "All Steps"] * len(x_draw)
    date_info = valid_time 
    df_scatter = pd.DataFrame(
        {"x_rmse": x_draw, "y_rmse": y_draw, "date_info": date_info}
    )

    fig_scatter = px.scatter(
        df_scatter,
        x="x_rmse",
        y="y_rmse",
        custom_data=["date_info"],
        opacity=0.35 if table_name == "grid_rmse_scatter" else 0.65,
        labels={
            "x_rmse": f"{exp_x} (mm/hr)",
            "y_rmse": f"{exp_y} (mm/hr)",
        },
        title=f"【Pred vs. Truth {mode_name} RMSE 對比】{exp_x} vs. {exp_y}",
    )

    fig_scatter.update_traces(
        hovertemplate=(
            "<b>時間/資料點:</b> %{customdata[0]}<br>"
            f"<b>{exp_x} RMSE:</b> %{{x:.4f}} mm/hr<br>"
            f"<b>{exp_y} RMSE:</b> %{{y:.4f}} mm/hr"
            "<extra></extra>"
        )
    )

    max_val = max(float(x_draw.max()), float(y_draw.max())) * 1.05
    fig_scatter.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_val,
        y1=max_val,
        line=dict(color="Red", dash="dash"),
    )
    fig_scatter.update_xaxes(range=[0, max_val])
    fig_scatter.update_yaxes(range=[0, max_val])

    time_label = f"有效時間: {valid_time}" if valid_time else "數據類型: 時間序列"
    anno_text = (
#       f"<b>{time_label}</b><br>"
        f"<b>Y > X 比例: {ratio_y_gt_x:.1f}%</b><br>"
        f"X > Y 比例: {100-ratio_y_gt_x:.1f}%"
    )
    fig_scatter.add_annotation(
        xref="paper",
        yref="paper",
        x=0.03,
        y=0.95,
        text=anno_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.85)",
        bordercolor="gray",
        borderwidth=1,
    )
    st.plotly_chart(fig_scatter)
  else:
    st.warning("⚠️ 該組實驗對比無有效散佈資料。")


# ==========================================
# 5. 側邊欄選單
# ==========================================
st.sidebar.title("🌧️ CorrDiff 評估儀表板")
tab_selection = st.sidebar.radio(
    "請選擇分析模式：",
    [
        "📊 全時間平均指標 (RMSE & FSS)",
        "🗺️ 跨實驗 RMSE 散佈圖比較",
        "🌍 空間場數據繪圖 (Spatial Plot)",
    ],
)
st.sidebar.markdown("---")


# ==========================================
# TAB 1: 全時間平均指標分析 (RMSE & FSS)
# ==========================================
if tab_selection == "📊 全時間平均指標 (RMSE & FSS)":
  st.title("📊 各實驗全時間平均指標 (Prediction vs. Truth)")

  df_metrics = load_all_time_metrics(DB_PATH)
  if df_metrics is None or df_metrics.empty:
    st.error(
        f"❌ 找不到資料庫檔案 `{DB_PATH}` 或無資料，請先執行預處理腳本。"
    )
    st.stop()

  all_exps = sorted(df_metrics["exp_name"].unique().tolist())
  selected_exps = st.sidebar.multiselect(
      "選擇要比較的實驗：", options=all_exps, default=all_exps
  )

  if not selected_exps:
    st.info("請至少選擇一個實驗。")
    st.stop()

  df_filtered = df_metrics[df_metrics["exp_name"].isin(selected_exps)]

  # 1. 總平均 RMSE
  st.subheader("1. 跨實驗總平均 RMSE 對比 (越低越好)")
  df_rmse_avg = (
      df_filtered.groupby("exp_name", as_index=False)["rmse"]
      .mean()
      .sort_values("rmse")
  )
  cols = st.columns(len(df_rmse_avg))
  best_rmse = df_rmse_avg["rmse"].min()

  for idx, row in df_rmse_avg.reset_index(drop=True).iterrows():
    with cols[idx]:
      if row["rmse"] == best_rmse:
        st.metric(f"🏆 {row['exp_name']} (最佳)", f"{row['rmse']:.4f} mm/hr")
      else:
        st.metric(
            row["exp_name"],
            f"{row['rmse']:.4f} mm/hr",
            delta=f"+{row['rmse'] - best_rmse:.4f}",
            delta_color="inverse",
        )

  fig_rmse = px.bar(
      df_rmse_avg, x='exp_name', y='rmse', color='exp_name', text_auto='.4f',
      title="Prediction vs. Truth 全時間平均 RMSE (mm/hr)",
      labels={'exp_name': '實驗名稱', 'rmse': '平均 RMSE (mm/hr)'}
  )
  fig_rmse.update_traces(textposition='outside')
  fig_rmse.update_layout(showlegend=False, yaxis_range=[0, df_rmse_avg['rmse'].max() * 1.25])
  st.plotly_chart(fig_rmse, width="stretch")

  st.markdown("---")

  # 2. FSS 分數圖表與熱力圖
  st.subheader("2. Fractional Skill Score (FSS) 平均表現 (越高越好)")
  fss_tab1, fss_tab2, fss_tab3 = st.tabs(
      ["🎯 特定門檻/視窗", "📈 FSS 隨門檻變化曲線", "🔥 FSS 矩陣熱力圖"]
  )

  with fss_tab1:
    c1, c2 = st.columns(2)
    with c1:
      selected_th = st.selectbox(
          "選擇降雨門檻 (mm/hr):",
          sorted(df_filtered["fss_threshold"].unique()),
          index=2,
      )
    with c2:
      selected_w = st.selectbox(
          "選擇視窗大小:",
          sorted(df_filtered["fss_window_size"].unique()),
          index=2,
      )

    df_fss_sub = (
        df_filtered[
            (df_filtered["fss_threshold"] == selected_th)
            & (df_filtered["fss_window_size"] == selected_w)
        ]
        .groupby("exp_name", as_index=False)["fss_score"]
        .mean()
    )
    fig_fss_bar = px.bar(
        df_fss_sub,
        x="exp_name",
        y="fss_score",
        color="exp_name",
        text_auto=".3f",
    )
    fig_fss_bar.update_layout(showlegend=False, yaxis_range=[0, 1.1])
    st.plotly_chart(fig_fss_bar)

  with fss_tab2:
    selected_w_tab2 = st.selectbox(
        "選擇空間視窗大小:",
        sorted(df_filtered["fss_window_size"].unique()),
        index=2,
        key="w_tab2",
    )
    df_fss_line = (
        df_filtered[df_filtered["fss_window_size"] == selected_w_tab2]
        .groupby(["exp_name", "fss_threshold"], as_index=False)["fss_score"]
        .mean()
    )

    fig_fss_line = px.line(
        df_fss_line,
        x="fss_threshold",
        y="fss_score",
        color="exp_name",
        markers=True,
        labels={
            "fss_threshold": "降雨門檻 Threshold (mm/hr)",
            "fss_score": "平均 FSS 分數",
        },
    )
    fig_fss_line.update_layout(yaxis_range=[0, 1.05])
    fig_fss_line.update_xaxes(type="category")
    st.plotly_chart(fig_fss_line)

  with fss_tab3:
    exp_for_hm = st.selectbox("選擇實驗：", selected_exps, key="hm_exp")
    df_hm = (
        df_filtered[df_filtered["exp_name"] == exp_for_hm]
        .groupby(["fss_threshold", "fss_window_size"], as_index=False)[
            "fss_score"
        ]
        .mean()
    )
    pivot_fss = df_hm.pivot(
        index="fss_window_size", columns="fss_threshold", values="fss_score"
    )

    fig_hm = px.imshow(
        pivot_fss, text_auto=".3f", color_continuous_scale="Viridis", zmin=0, zmax=1
    )
    fig_hm.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_hm)


# ==========================================
# TAB 2: 跨實驗 RMSE 散佈圖比較 (格點 & 時間維度)
# ==========================================
elif tab_selection == "🗺️ 跨實驗 RMSE 散佈圖比較":
  st.title("🗺️ 跨實驗 RMSE 散佈圖比較 (Scatter Plot)")

  sub_tab_grid, sub_tab_time = st.tabs([
      "🗺️ 格點層級散佈圖 (grid_rmse_scatter)",
      "⏱️ 時間層級散佈圖 (time_rmse_scatter)",
  ])

  with sub_tab_grid:
    render_scatter_plot("grid_rmse_scatter", "格點層級 (Grid)")

  with sub_tab_time:
    render_scatter_plot("time_rmse_scatter", "時間層級 (Time)")


# ==========================================
# TAB 3: 空間場數據繪圖 (Spatial Plot)
# ==========================================
elif tab_selection == "🌍 空間場數據繪圖 (Spatial Plot)":
  st.title("🌍 空間場數據繪圖 (Target / Input / Prediction 對比)")

  lats, lons, land_mask, ter = load_wrf_grid_coords(GRID_COORDS_NC)
  if lats is None:
    st.error(f"❌ 找不到座標檔案 `{GRID_COORDS_NC}`，請確認檔案位置。")
    st.stop()

  exp_list = list(EXPERIMENT_NC_MAP.keys())
  selected_exp = st.sidebar.selectbox(
      "選擇實驗組別：", options=exp_list, index=0
  )
  nc_path = EXPERIMENT_NC_MAP[selected_exp]

  if not os.path.exists(nc_path):
    st.error(f"❌ 找不到檔案：`{nc_path}`\n請檢查相對路徑！")
    st.stop()

  times = get_nc_time_list(nc_path)

  col_var, col_time = st.columns([1, 2])
  with col_var:
    var_opts = {
        "降雨量 (precipitation)": "precipitation",
        "2米氣溫": "temperature_2m",
    }
    var_name = var_opts[st.selectbox("選擇變數：", list(var_opts.keys()))]
  with col_time:
    selected_time_str = st.selectbox("選擇日期/時間：", times, index=0)
    time_idx = times.index(selected_time_str)

  truth_2d, pred_2d, input_2d = load_spatial_slice(nc_path, time_idx, var_name)

  fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
  fig.suptitle(
      f"{var_name} - {selected_time_str} ({selected_exp})", fontsize=18, y=0.96
  )

  lon_min, lon_max = lons.min(), lons.max()
  lat_min, lat_max = lats.min(), lats.max()
  extent = [lon_min, lon_max, lat_min, lat_max]

  for idx, ax in enumerate(axes):
    data = [truth_2d, input_2d, pred_2d][idx]
    title = ["Target (TReAD)", "Input (ERA5)", "CorrDiff (Reg)"][idx]

    im = ax.imshow(
        data,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap=cwb_cmap if var_name == "precipitation" else "turbo",
        norm=cwb_norm if var_name == "precipitation" else None,
    )

    if land_mask is not None:
      ax.contour(
          lons, lats, land_mask, levels=[0.5], colors="black", linewidths=1.0
      )

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("經度 (Longitude)")
    if idx == 0:
      ax.set_ylabel("緯度 (Latitude)")

  fig.subplots_adjust(right=0.84, wspace=0.2)
  cbar = fig.colorbar(
      im,
      cax=fig.add_axes([0.86, 0.18, 0.02, 0.65]),
      extend="max" if var_name == "precipitation" else "neither",
  )
  if var_name == "precipitation":
    cbar.set_ticks(PRECIP_BOUNDS)
    cbar.set_label("降雨量 (mm/hr)", fontsize=12)

  st.pyplot(fig)
  plt.close(fig)  # 釋放繪圖資源
