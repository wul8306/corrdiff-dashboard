import os
import sqlite3
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter


# ==============================================================================
# 一、核心計算模組 (Core Computation Module)
# ==============================================================================

def calculate_fss(obs_2d: np.ndarray, pred_2d: np.ndarray, threshold: float, window_size: int) -> float:
    """
    使用 2D 滑動視窗 (Uniform Filter) 高效計算 Fractional Skill Score (FSS)

    :param obs_2d: 觀測值 2D 陣列 (無 NaN)
    :param pred_2d: 預測值 2D 陣列 (無 NaN)
    :param threshold: 降雨強度門檻值 (mm/hr)
    :param window_size: 空間視窗大小 (如 3, 5, 7, 9...)
    :return: FSS 分數 (0.0 ~ 1.0)
    """
    # 建立二值化遮罩
    obs_binary = (obs_2d >= threshold).astype(np.float32)
    pred_binary = (pred_2d >= threshold).astype(np.float32)

    # 利用 2D 均值濾波器快速計算視窗內發生事件的比例 (Fraction)
    obs_frac = uniform_filter(obs_binary, size=window_size, mode='constant', cval=0.0)
    pred_frac = uniform_filter(pred_binary, size=window_size, mode='constant', cval=0.0)

    # 計算 MSE 與參考 MSE
    mse = np.mean((obs_frac - pred_frac) ** 2)
    mse_ref = np.mean(obs_frac ** 2 + pred_frac ** 2)

    if mse_ref == 0:
        return 1.0 if mse == 0 else 0.0

    fss = 1.0 - (mse / mse_ref)
    return float(fss)


def calculate_grid_rmse(da_truth: xr.DataArray, da_pred_mean: xr.DataArray) -> np.ndarray:
    """
    計算空間中每一個獨立網格點 (Grid-level) 的平均 RMSE 陣列 (展平成 1D)

    :return: 展平後的一維 float32 陣列
    """
    sq_err = (da_pred_mean - da_truth) ** 2

    if 'time' in da_truth.dims and da_truth.sizes['time'] > 0:
        if np.issubdtype(da_truth['time'].dtype, np.datetime64):
            try:
                # 每日 Resampling 取均方根再取全時段平均
                daily_rmse = np.sqrt(sq_err.resample(time='1D').mean(dim='time'))
                mean_rmse = daily_rmse.mean(dim='time')
            except Exception:
                mean_rmse = np.sqrt(sq_err.mean(dim='time'))
        else:
            mean_rmse = np.sqrt(sq_err.mean(dim='time'))

        grid_rmse = mean_rmse.values.flatten().astype(np.float32)
    else:
        grid_rmse = np.sqrt(sq_err.values).flatten().astype(np.float32)

    return grid_rmse


def calculate_rmse_t_series(da_truth: xr.DataArray, da_pred_mean: xr.DataArray) -> tuple[np.ndarray, list[str]]:
    """
    計算每個時間步長 (Time Step) 的全空間區域 RMSE (rmse_t) 與時間標籤

    :return: (rmse_t_array, times_str_list)
    """
    raw_times = da_truth['time'].values
    if np.issubdtype(raw_times.dtype, np.datetime64):
        time_series = pd.to_datetime(raw_times)
        times_str = [t.strftime('%Y-%m-%d %H:%M') for t in time_series]
    else:
        base_date = pd.Timestamp("2016-01-01 00:00:00")
        times_str = [(base_date + pd.Timedelta(days=float(t))).strftime('%Y-%m-%d %H:%M') for t in raw_times]

    rmse_t_list = []
    n_times = len(times_str)

    # 逐時間點計算全空間 RMSE
    for t_idx in range(n_times):
        truth_2d = da_truth.isel(time=t_idx).values if 'time' in da_truth.dims else da_truth.values
        pred_2d = da_pred_mean.isel(time=t_idx).values if 'time' in da_pred_mean.dims else da_pred_mean.values

        valid_mask = ~(np.isnan(truth_2d) | np.isnan(pred_2d))
        if np.any(valid_mask):
            sq_diff = (pred_2d[valid_mask] - truth_2d[valid_mask]) ** 2
            rmse_val = float(np.sqrt(np.mean(sq_diff)))
        else:
            rmse_val = np.nan

        rmse_t_list.append(rmse_val)

    return np.array(rmse_t_list, dtype=np.float32), times_str


# ==============================================================================
# 二、綜合指標計算與評估 (Full Metrics Evaluation)
# ==============================================================================

def compute_experiment_metrics(nc_path: str, var_name: str = "precipitation",
                               fss_thresholds: list = None, fss_windows: list = None) -> dict:
    """
    對單一實驗 NetCDF 進行完整指標計算 (包含 Grid-RMSE、RMSE_t、與跨 48 種組合的 FSS)
    """
    if fss_thresholds is None:
        fss_thresholds = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0]
    if fss_windows is None:
        fss_windows = [3, 5, 7, 9, 15, 20]

    try:
        ds_truth = xr.open_dataset(nc_path, group="truth")
        ds_pred = xr.open_dataset(nc_path, group="prediction")
    except Exception:
        ds = xr.open_dataset(nc_path)
        ds_truth, ds_pred = ds, ds

    da_truth = ds_truth[var_name]
    da_pred = ds_pred[var_name]

    if "ensemble" in da_pred.dims:
        da_pred = da_pred.mean(dim="ensemble")

    # 1. 格點級 RMSE (Grid-level)
    grid_rmse_arr = calculate_grid_rmse(da_truth, da_pred)

    # 2. 時間軸 RMSE (Time-level)
    rmse_t_arr, times_str = calculate_rmse_t_series(da_truth, da_pred)

    # 3. 多尺度 FSS 計算
    time_fss_dict = {(th, w): [] for th in fss_thresholds for w in fss_windows}
    n_times = len(times_str)

    for t_idx in range(n_times):
        truth_2d = da_truth.isel(time=t_idx).values if 'time' in da_truth.dims else da_truth.values
        pred_2d = da_pred.isel(time=t_idx).values if 'time' in da_pred.dims else da_pred.values

        valid_mask = ~(np.isnan(truth_2d) | np.isnan(pred_2d))
        if not np.any(valid_mask):
            continue

        truth_filled = np.nan_to_num(truth_2d, nan=0.0)
        pred_filled = np.nan_to_num(pred_2d, nan=0.0)

        for th in fss_thresholds:
            for w in fss_windows:
                fss_val = calculate_fss(truth_filled, pred_filled, th, w)
                time_fss_dict[(th, w)].append(fss_val)

    ds_truth.close()
    ds_pred.close()

    # 整理平均 FSS
    avg_fss_dict = {key: float(np.mean(val)) if val else 0.0 for key, val in time_fss_dict.items()}

    return {
        "grid_rmse": grid_rmse_arr,
        "rmse_t": rmse_t_arr,
        "times_str": times_str,
        "avg_fss": avg_fss_dict
    }


# ==============================================================================
# 三、預計算與 SQLite 資料庫導出模組 (Precomputation & Database Export)
# ==============================================================================

def init_database(db_path: str):
    """建立 SQLite 資料庫及對應資料表"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 跨實驗【格點級】RMSE 散佈資料表 (Grid-level Scatter)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS grid_rmse_scatter (
            exp_x TEXT,
            exp_y TEXT,
            var_name TEXT,
            valid_time TEXT,
            ratio_y_gt_x REAL,
            x_values BLOB,
            y_values BLOB,
            PRIMARY KEY (exp_x, exp_y, var_name)
        )
    """)

    # 2. 跨實驗【時間軸級】RMSE 散佈資料表 (Time-level Scatter)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS time_rmse_scatter (
            exp_x TEXT,
            exp_y TEXT,
            var_name TEXT,
            valid_time TEXT,
            ratio_y_gt_x REAL,
            x_values BLOB,
            y_values BLOB,
            PRIMARY KEY (exp_x, exp_y, var_name)
        )
    """)

    # 3. 摘要統計結果表 (Summary Metrics)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summary_metrics (
            exp_name TEXT,
            file_name TEXT,
            time_label TEXT,
            comp_type TEXT,
            var_name TEXT,
            avg_rmse REAL,
            threshold REAL,
            window INTEGER,
            avg_fss REAL,
            PRIMARY KEY (exp_name, file_name, var_name, threshold, window)
        )
    """)

    conn.commit()
    conn.close()


def run_precomputations(exp_nc_map: dict[str, str], db_path: str = "corrdiff_allmetrics.db", var_name: str = "precipitation"):
    """
    執行完整預計算，將 Grid-level RMSE、Time-level RMSE 與 Summary Metrics 寫入 SQLite
    """
    init_database(db_path)

    # 步驟 1: 計算所有實驗的指標資料
    exp_results = {}
    for exp_name, nc_path in exp_nc_map.items():
        if os.path.exists(nc_path):
            print(f"⏳ 正在分析實驗與計算指標: [{exp_name}] -> {nc_path}")
            exp_results[exp_name] = compute_experiment_metrics(nc_path, var_name=var_name)
        else:
            print(f"⚠️ 找不到檔案，跳過: {nc_path}")

    exp_names = list(exp_results.keys())
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 步驟 2: 寫入摘要統計表 (summary_metrics)
        print("\n📝 正在寫入 Summary Metrics (FSS & Overall RMSE)...")
        for exp_name, data in exp_results.items():
            avg_rmse = float(np.nanmean(data['rmse_t']))
            nc_filename = os.path.basename(exp_nc_map[exp_name])

            for (th, w), avg_fss in data['avg_fss'].items():
                cursor.execute("""
                    INSERT OR REPLACE INTO summary_metrics
                    (exp_name, file_name, time_label, comp_type, var_name, avg_rmse, threshold, window, avg_fss)
                    VALUES (?, ?, 'ALL_TIME_AVG', 'pred_vs_truth', ?, ?, ?, ?, ?)
                """, (exp_name, nc_filename, var_name, avg_rmse, th, w, avg_fss))

        # 步驟 3: 兩兩配對寫入散佈圖 (Grid-level & Time-level)
        print("\n🔗 正在計算兩兩配對比對 (Pairwise Comparisons)...")
        for i in range(len(exp_names)):
            for j in range(i + 1, len(exp_names)):
                exp_x, exp_y = exp_names[i], exp_names[j]

                print(f"--- A. 寫入 格點級 (Grid-level) 散佈圖 ---")
                gx, gy = exp_results[exp_x]['grid_rmse'], exp_results[exp_y]['grid_rmse']
                g_valid = ~(np.isnan(gx) | np.isnan(gy))
                gx_v, gy_v = gx[g_valid], gy[g_valid]

                if len(gx_v) > 0:
                    g_ratio = float(np.mean(gy_v > gx_v) * 100.0)
                    cursor.execute("""
                        INSERT OR REPLACE INTO grid_rmse_scatter
                        (exp_x, exp_y, var_name, valid_time, ratio_y_gt_x, x_values, y_values)
                        VALUES (?, ?, ?, 'GRID_POINTS', ?, ?, ?)
                    """, (exp_x, exp_y, var_name, g_ratio, gx_v.tobytes(), gy_v.tobytes()))
                    print(f"  ├─ [Grid Scatter] {exp_x} vs {exp_y}: {len(gx_v)} 個網格點, Y > X 比例: {g_ratio:.2f}%")

                print(f"--- B. 寫入 時間軸級 (Time-level) 散佈圖 ---")
                tx, ty = exp_results[exp_x]['rmse_t'], exp_results[exp_y]['rmse_t']
                times_str = exp_results[exp_x]['times_str']
                t_valid = ~(np.isnan(tx) | np.isnan(ty))
                tx_v, ty_v = tx[t_valid], ty[t_valid]
                valid_times_list = [times_str[k] for k in range(len(times_str)) if t_valid[k]]

                if len(tx_v) > 0:
                    t_ratio = float(np.mean(ty_v > tx_v) * 100.0)
                    time_json = json.dumps(valid_times_list)
                    cursor.execute("""
                        INSERT OR REPLACE INTO time_rmse_scatter
                        (exp_x, exp_y, var_name, valid_time, ratio_y_gt_x, x_values, y_values)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (exp_x, exp_y, var_name, time_json, t_ratio, tx_v.tobytes(), ty_v.tobytes()))
                    print(f"  └─ [Time Scatter] {exp_x} vs {exp_y}: {len(tx_v)} 個時間點, Y > X 比例: {t_ratio:.2f}%")

        conn.commit()
    finally:
        conn.close()

    print("\n🎉 所有人預計算指標與散佈圖資料皆已成功更新至 SQLite 資料庫！")


# ==============================================================================
# 四、主程式執行進入點
# ==============================================================================

if __name__ == "__main__":
    # 1. 定義實驗名稱與檔名對照表
    EXPERIMENT_NC_MAP = {
        "P2P": "output/P2P25Y/output_0_all.nc",
        "CP2P": "output/CP2P25Y/output_0_all.nc",
    }

    # 2. 執行計算與導出
    DB_FILE = "corrdiff_allmetrics.db"
    run_precomputations(EXPERIMENT_NC_MAP, db_path=DB_FILE, var_name="precipitation")
