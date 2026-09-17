"""
NetCDF Groups 指標與跨實驗格點 RMSE 預計算腳本
======================================================
計算:
1. 時間軸指標: Prediction (Ensemble Mean) vs. Truth 的 RMSE 與 FSS
2. 空間格點指標: 跨實驗組合 (X vs. Y) 的空間格點 RMSE 散佈圖數據 (BLOB 壓縮儲存)

將所有數據寫入 SQLite 資料庫 (corrdiff_allmetrics.db) 供 Streamlit 快速展示。
"""

import os
import glob
import sqlite3
import numpy as np
import pandas as pd  # 確保導入 pandas (用於時間解析)
import xarray as xr
from scipy.ndimage import uniform_filter
from itertools import combinations

DB_FILE = "corrdiff_allmetrics.db"

# ==========================================
# 1. 資料庫初始化
# ==========================================
def init_db(db_path=DB_FILE):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # (A) 時間軸區域指標表 (RMSE / FSS)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiment_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exp_name TEXT NOT NULL,
        file_name TEXT NOT NULL,
        valid_time TEXT,
        comparison_type TEXT, -- 固定為 'pred_vs_truth'
        variable_name TEXT,   -- 'precipitation' 等
        rmse REAL,
        fss_threshold REAL,
        fss_window_size INTEGER,
        fss_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(exp_name, file_name, valid_time, comparison_type, variable_name, fss_threshold, fss_window_size)
    );
    """)

    # (B) 跨實驗格點 RMSE 散佈圖資料表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS grid_rmse_scatter (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exp_x TEXT NOT NULL,
        exp_y TEXT NOT NULL,
        valid_time TEXT NOT NULL,
        var_name TEXT NOT NULL,
        ratio_y_gt_x REAL,
        x_values BLOB, -- 儲存 float32 陣列 (Numpy bytes)
        y_values BLOB, -- 儲存 float32 陣列 (Numpy bytes)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(exp_x, exp_y, valid_time, var_name)
    );
    """)

    conn.commit()
    conn.close()


# ==========================================
# 2. 核心計算模組 (FSS & 時間軸 / 格點指標)
# ==========================================
def calculate_fss(obs_array, pred_array, threshold, window_size):
    obs_binary = (obs_array >= threshold).astype(float)
    pred_binary = (pred_array >= threshold).astype(float)

    I_o = uniform_filter(obs_binary, size=window_size, mode='constant', cval=0.0)
    I_p = uniform_filter(pred_binary, size=window_size, mode='constant', cval=0.0)

    mse = np.mean((I_p - I_o) ** 2)
    ref_mse = np.mean(I_p ** 2) + np.mean(I_o ** 2)

    if ref_mse == 0:
        return 1.0 if mse == 0 else 0.0
    return float(np.clip(1.0 - (mse / ref_mse), 0.0, 1.0))


def process_time_series_metrics(nc_path, exp_name, var_name="precipitation"):
    """
    計算單一 NC 檔案中的 Prediction vs. Truth 指標，
    包含：
    1. 時間軸平均指標 (records: RMSE, FSS)
    2. 全空間格點 RMSE 陣列 (grid_rmse)
    3. valid_time 時間標籤字串 (valid_time_str)
    """
    results = []
    grid_rmse = None
    valid_time_str = "N/A"

    try:
        ds_truth = xr.open_dataset(nc_path, group="truth")
        ds_pred = xr.open_dataset(nc_path, group="prediction")
    except Exception as e:
        print(f"[ERROR] 讀取 Group 失敗 {nc_path}: {e}")
        return results, grid_rmse, valid_time_str

    print(f"  └─ Read finish : {nc_path}")

    if var_name not in ds_truth:
        print(f"[WARNING] 檔案內找不到變數 {var_name}")
        ds_truth.close()
        ds_pred.close()
        return results, grid_rmse, valid_time_str

    da_truth = ds_truth[var_name]

    # 對 Ensemble 取平均
    if "ensemble" in ds_pred[var_name].dims:
        da_pred_mean = ds_pred[var_name].mean(dim="ensemble")
    else:
        da_pred_mean = ds_pred[var_name]

    # ------------------------------------------
    # A. 解析並格式化 valid_time
    # ------------------------------------------
    if 'time' in da_truth.coords and len(da_truth['time']) > 0:
        try:
            if np.issubdtype(da_truth['time'].dtype, np.datetime64):
                t_vals = pd.to_datetime(da_truth['time'].values)
                unique_dates = sorted(list(set(t_vals.strftime('%Y-%m-%d'))))
            else:
                base_date = pd.Timestamp("2016-01-01 00:00:00")
                t_series = [base_date + pd.Timedelta(days=float(t)) for t in da_truth['time'].values]
                unique_dates = sorted(list(set([t.strftime('%Y-%m-%d') for t in t_series])))

            if len(unique_dates) == 1:
                valid_time_str = unique_dates[0]
            elif len(unique_dates) <= 3:
                valid_time_str = ", ".join(unique_dates)
            else:
                valid_time_str = f"{unique_dates[0]} ~ {unique_dates[-1]}"
        except Exception:
            valid_time_str = "ALL_TIME"

    # ------------------------------------------
    # B. 計算空間格點 (Grid-level) RMSE 陣列
    # 計算資料集中「每一個空間網格點（Grid Cell）」的平均 RMSE（均方根誤差），並將結果展平成一維陣列（1D Array）。
    # 使用 .flatten() 展平成一維陣列（變成長度 208 x 208 = 43264 的一維陣列）
    # ------------------------------------------
    sq_err = (da_pred_mean - da_truth) ** 2
    if 'time' in da_truth.dims and da_truth.sizes['time'] > 0:
        if np.issubdtype(da_truth['time'].dtype, np.datetime64):
            try:
                daily_rmse = np.sqrt(sq_err.resample(time='1D').mean(dim='time'))
                mean_rmse = daily_rmse.mean(dim='time')
            except Exception:
                mean_rmse = np.sqrt(sq_err.mean(dim='time'))
        else:
            mean_rmse = np.sqrt(sq_err.mean(dim='time'))
        grid_rmse = mean_rmse.values.flatten().astype(np.float32)
    else:
        grid_rmse = np.sqrt(sq_err.values).flatten().astype(np.float32)

    # ------------------------------------------
    # C. 計算時間軸指標 (RMSE & FSS)
    # ------------------------------------------
    fss_thresholds = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0]
    fss_windows = [3, 5, 7, 9, 15, 20]

    times = da_truth['time'].values if 'time' in da_truth.coords else [None]

    time_rmses = []
    time_fss_dict = {(th, w): [] for th in fss_thresholds for w in fss_windows}

    for t_idx, t_val in enumerate(times):
        truth_2d = da_truth.isel(time=t_idx).values if 'time' in da_truth.coords else da_truth.values
        pred_2d = da_pred_mean.isel(time=t_idx).values if 'time' in da_pred_mean.coords else da_pred_mean.values

        valid_mask = ~(np.isnan(truth_2d) | np.isnan(pred_2d))
        if not np.any(valid_mask):
            continue

        rmse_t = float(np.sqrt(np.mean((pred_2d[valid_mask] - truth_2d[valid_mask]) ** 2)))
        time_rmses.append(rmse_t)

        truth_filled = np.nan_to_num(truth_2d, nan=0.0)
        pred_filled = np.nan_to_num(pred_2d, nan=0.0)

        for th in fss_thresholds:
            for w in fss_windows:
                fss_val = calculate_fss(truth_filled, pred_filled, th, w)
                time_fss_dict[(th, w)].append(fss_val)

    ds_truth.close()
    ds_pred.close()

    if time_rmses:
        avg_rmse = float(np.mean(time_rmses))
        comp_type = "pred_vs_truth"
        time_label = "ALL_TIME_AVG"

        for (th, w), fss_list in time_fss_dict.items():
            avg_fss = float(np.mean(fss_list)) if fss_list else 0.0
            results.append((
                exp_name,
                os.path.basename(nc_path),
                time_label,
                comp_type,
                var_name,
                avg_rmse,
                th,
                w,
                avg_fss
            ))

    return results, grid_rmse, valid_time_str


def save_time_metrics_to_db(results, db_path=DB_FILE):
    if not results:
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR REPLACE INTO experiment_metrics
    (exp_name, file_name, valid_time, comparison_type, variable_name, rmse, fss_threshold, fss_window_size, fss_score)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, results)
    conn.commit()
    conn.close()


# ==========================================
# 3. 跨實驗格點 (Grid-level) RMSE 預計算模組
# ==========================================
def precompute_grid_rmse_scatter(rmse_dict, time_label_dict, var_name="precipitation", max_points=50000, db_path=DB_FILE):
    """直接使用 process_time_series_metrics 產生的格點 RMSE 與 valid_time，進行兩兩實驗交叉計算並寫入 DB"""
    print("\n[INFO] 開始預計算跨實驗格點 (Grid-level) RMSE 散佈圖數據...")

    exp_names = list(rmse_dict.keys())
    if len(exp_names) < 2:
        print("  └─ [NOTICE] 實驗數量少於 2 個，無法進行兩兩比對。")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 兩兩組合進行交叉比對
    for exp_x, exp_y in combinations(exp_names, 2):
        print(f"  └─ 交叉計算: {exp_x} (X軸) vs. {exp_y} (Y軸)")

        rmse_x = rmse_dict[exp_x]
        rmse_y = rmse_dict[exp_y]

        if rmse_x is None or rmse_y is None:
            continue

        # 以 X 軸實驗的時間標籤為準
        valid_time_label = time_label_dict.get(exp_x, "N/A")

        valid_mask = ~(np.isnan(rmse_x) | np.isnan(rmse_y))
        x_vals = rmse_x[valid_mask].astype(np.float32)
        y_vals = rmse_y[valid_mask].astype(np.float32)

        if len(x_vals) == 0:
            continue

        # 計算全域精準 Y > X 比例
        ratio_y_gt_x = float((np.sum(y_vals > x_vals) / len(x_vals)) * 100.0)

        # 隨機抽樣以維持繪圖效能
        if len(x_vals) > max_points:
            print(f"     隨機抽樣 {max_points} 點以維持繪圖效能")
            idx = np.random.choice(len(x_vals), size=max_points, replace=False)
            x_save, y_save = x_vals[idx], y_vals[idx]
        else:
            print(f"     無須隨機抽樣 (共 {len(x_vals)} 點)")
            x_save, y_save = x_vals, y_vals

        # 寫入 SQLite (包含 valid_time 資訊)
        cursor.execute("""
        INSERT OR REPLACE INTO grid_rmse_scatter
        (exp_x, exp_y, valid_time, var_name, ratio_y_gt_x, x_values, y_values)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (exp_x, exp_y, valid_time_label, var_name, ratio_y_gt_x, x_save.tobytes(), y_save.tobytes()))

    conn.commit()
    conn.close()
    print("  └─ 跨實驗格點 RMSE 數據已成功寫入資料庫！")


# ==========================================
# 4. 主執行程序
# ==========================================
if __name__ == "__main__":
    init_db(DB_FILE)

    base_dir = "output"
    experiments_patterns = {
        "P2P": os.path.join(base_dir, "P2P25Y/output_0_all.nc"),
        "CP2P": os.path.join(base_dir, "CP2P25Y/output_0_all.nc"),
    }

    experiments_files = {}
    rmse_dict = {}
    time_label_dict = {}

    # 階段 1：計算各實驗時間軸與格點指標 (Pred vs Truth)
    for exp_name, path_pattern in experiments_patterns.items():
        nc_files = sorted(glob.glob(path_pattern))
        experiments_files[exp_name] = nc_files
        print(f"\n[INFO] 開始處理實驗時間軸與格點指標：{exp_name}，共找到 {len(nc_files)} 個檔案")

        for nc_file in nc_files:
            print(f"  └─ 處理檔案: {os.path.basename(nc_file)}")
            records, grid_rmse, valid_time_str = process_time_series_metrics(
                nc_file, exp_name=exp_name, var_name="precipitation"
            )

            # 儲存時間軸指標至 DB
            if records:
                save_time_metrics_to_db(records, db_path=DB_FILE)
                print(f"     已寫入 {len(records)} 筆 Pred vs Truth 指標數據")

            # 快取格點 RMSE 與時間標籤，供階段 2 使用
            if grid_rmse is not None:
                rmse_dict[exp_name] = grid_rmse
                time_label_dict[exp_name] = valid_time_str

    # 階段 2：預計算跨實驗格點 RMSE 散佈圖數據 (直接重用階段 1 結果)
    precompute_grid_rmse_scatter(
        rmse_dict, time_label_dict, var_name="precipitation", max_points=50000, db_path=DB_FILE
    )

    print("\n[SUCCESS] 所有預計算流程順利完成！")
