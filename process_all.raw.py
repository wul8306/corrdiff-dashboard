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
        var_name TEXT NOT NULL,
        ratio_y_gt_x REAL,
        x_values BLOB, -- 儲存 float32 陣列 (Numpy bytes)
        y_values BLOB, -- 儲存 float32 陣列 (Numpy bytes)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(exp_x, exp_y, var_name)
    );
    """)
    
    conn.commit()
    conn.close()


# ==========================================
# 2. 核心計算模組 (FSS & 時間軸指標)
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
    """計算單一 NC 檔案中的 Prediction vs. Truth 指標，並輸出全時間軸平均 (RMSE, FSS)"""
    results = []

    try:
        ds_truth = xr.open_dataset(nc_path, group="truth")
        ds_pred = xr.open_dataset(nc_path, group="prediction")
    except Exception as e:
        print(f"[ERROR] 讀取 Group 失敗 {nc_path}: {e}")
        return results

    if var_name not in ds_truth:
        print(f"[WARNING] 檔案內找不到變數 {var_name}")
        return results

    da_truth = ds_truth[var_name]

    # 對 Ensemble 取平均
    if "ensemble" in ds_pred[var_name].dims:
        da_pred_mean = ds_pred[var_name].mean(dim="ensemble")
    else:
        da_pred_mean = ds_pred[var_name]

    fss_thresholds = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0]
    fss_windows = [3, 5, 7, 9, 15, 20]

    times = da_truth['time'].values if 'time' in da_truth.coords else [None]

    # 暫存每一個時間步的純量結果
    time_rmses = []
    # 建立字典結構儲存各 (th, w) 組合的所有時間點 FSS： {(th, w): [fss_t1, fss_t2, ...]}
    time_fss_dict = {(th, w): [] for th in fss_thresholds for w in fss_windows}

    for t_idx, t_val in enumerate(times):
        truth_2d = da_truth.isel(time=t_idx).values if 'time' in da_truth.coords else da_truth.values
        pred_2d = da_pred_mean.isel(time=t_idx).values if 'time' in da_pred_mean.coords else da_pred_mean.values

        valid_mask = ~(np.isnan(truth_2d) | np.isnan(pred_2d))
        if not np.any(valid_mask):
            continue

        # 1. 計算該時間步的 RMSE 並存入列表
        rmse_t = float(np.sqrt(np.mean((pred_2d[valid_mask] - truth_2d[valid_mask]) ** 2)))
        time_rmses.append(rmse_t)

        # 2. 計算該時間步的各 FSS 並存入列表
        truth_filled = np.nan_to_num(truth_2d, nan=0.0)
        pred_filled = np.nan_to_num(pred_2d, nan=0.0)

        for th in fss_thresholds:
            for w in fss_windows:
                fss_val = calculate_fss(truth_filled, pred_filled, th, w)
                time_fss_dict[(th, w)].append(fss_val)

    ds_truth.close()
    ds_pred.close()

    # 若沒有任何有效時間步，直接返回空結果
    if not time_rmses:
        return results

    # 3. 計算全時間步的平均值 (Time-averaged)
    avg_rmse = float(np.mean(time_rmses))
    comp_type = "pred_vs_truth"
    time_label = "ALL_TIME_AVG"  # 表示此筆紀錄代表全時間軸平均

    for (th, w), fss_list in time_fss_dict.items():
        avg_fss = float(np.mean(fss_list)) if fss_list else 0.0
        results.append((
            exp_name,
            os.path.basename(nc_path),
            time_label,     # valid_time 欄位標示為平均
            comp_type,
            var_name,
            avg_rmse,       # 全時間軸平均 RMSE
            th,
            w,
            avg_fss         # 全時間軸平均 FSS
        ))

    return results


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
# 3. 跨實驗格點 (Grid-level) RMSE 計算模組
# ==========================================
def get_grid_rmse_from_nc(nc_path, var_name="precipitation"):
    """計算單一 NC 檔案全空間每個格點在時間軸上的 pred_vs_truth RMSE 陣列"""
    ds_truth = xr.open_dataset(nc_path, group="truth")
    ds_pred = xr.open_dataset(nc_path, group="prediction")

    da_truth = ds_truth[var_name]

    if "ensemble" in ds_pred[var_name].dims:
        da_pred = ds_pred[var_name].mean(dim="ensemble")
    else:
        da_pred = ds_pred[var_name]

# 計算殘差平方
    sq_err = (da_pred - da_truth) ** 2

    if 'time' in da_truth.dims and da_truth.sizes['time'] > 0:
        # 檢查 time 座標是否為 DatetimeIndex (可進行日期的 resample)
        is_datetime = np.issubdtype(da_truth['time'].dtype, np.datetime64)

        if is_datetime:
            # 1. 有日期資訊：先依據「天 ('1D')」取平均並開根號得到每日 RMSE，再對所有天數取平均
            try:
                daily_rmse = np.sqrt(sq_err.resample(time='1D').mean(dim='time'))
                mean_rmse = daily_rmse.mean(dim='time')
            except Exception:
                # 萬一 resample 失敗，降級為直接對全時間軸取平均
                mean_rmse = np.sqrt(sq_err.mean(dim='time'))
        else:
            # 2. 無日期資訊 (如純索引 0, 1, 2...)：直接對整條 time 維度取平均並開根號
            mean_rmse = np.sqrt(sq_err.mean(dim='time'))

        grid_rmse = mean_rmse.values.flatten()
    else:
        grid_rmse = np.sqrt(sq_err.values).flatten()


    ds_truth.close()
    ds_pred.close()
    return grid_rmse

def get_nc_time_list(nc_path):
    """解析相對時間天數並轉為 YYYY-MM-DD HH:MM"""
    if not os.path.exists(nc_path):
        return None
    ds = xr.open_dataset(nc_path)
    try:
        if np.issubdtype(ds['time'].dtype, np.datetime64):
            time_series = pd.to_datetime(ds['time'].values)
        else:
            base_date = pd.Timestamp("2016-01-01 00:00:00")
            time_series = [base_date + pd.Timedelta(days=float(t)) for t in ds['time'].values]
        times = [t.strftime('%Y-%m-%d %H:%M') for t in time_series]
    except Exception:
        times = [f"Step {idx} (Day Offset: {t})" for idx, t in enumerate(ds['time'].values)]

    ds.close()
    return times


def precompute_grid_rmse_scatter(experiments_all_files, var_name="precipitation", max_points=5000000, db_path=DB_FILE):
    """計算任意兩兩實驗組合的格點 RMSE，抽樣後轉為 BLOB 寫入 DB"""
    print("\n[INFO] 開始預計算跨實驗格點 (Grid-level) RMSE 散佈圖數據...")

    
    # 針對代表性檔案 (例如 output_0_all.nc) 讀取格點數據
    rmse_dict = {}
    for exp_name, nc_files in experiments_all_files.items():
        if not nc_files:
            continue
        # 優先找 output_0_all.nc，若無則使用第一個找到的 nc 檔
        target_nc = next((f for f in nc_files if "output_0_all.nc" in f), nc_files[0])
        print(f"  └─ 讀取格點資料 ({exp_name}): {os.path.basename(target_nc)}")
        rmse_dict[exp_name] = get_grid_rmse_from_nc(target_nc, var_name)

    exp_names = list(rmse_dict.keys())
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 兩兩組合進行交叉比對
    for exp_x, exp_y in combinations(exp_names, 2):
        print(f"  └─ 交叉計算: {exp_x} (X軸) vs. {exp_y} (Y軸)")
        
        rmse_x = rmse_dict[exp_x]
        rmse_y = rmse_dict[exp_y]

        valid_mask = ~(np.isnan(rmse_x) | np.isnan(rmse_y))
        x_vals = rmse_x[valid_mask].astype(np.float32)
        y_vals = rmse_y[valid_mask].astype(np.float32)

        if len(x_vals) == 0:
            continue

        # 計算全域精準 Y > X 比例
        ratio_y_gt_x = float((np.sum(y_vals > x_vals) / len(x_vals)) * 100.0)

        # 隨機抽樣 50,000 點以維持繪圖效能
        if len(x_vals) > max_points:
            print(f" 隨機抽樣 {max_points} 點以維持繪圖效能")
            idx = np.random.choice(len(x_vals), size=max_points, replace=False)
            x_save, y_save, date_values = x_vals[idx], y_vals[idx], date_values[idx]
        else:
            print(f" No 隨機抽樣")
            x_save, y_save, date_values = x_vals, y_vals, date_values

        # 寫入 SQLite (使用 tobytes 轉為二進位 BLOB)
        cursor.execute("""
        INSERT OR REPLACE INTO grid_rmse_scatter 
        (exp_x, exp_y, var_name, ratio_y_gt_x, x_values, y_values)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (exp_x, exp_y, var_name, ratio_y_gt_x, x_save.tobytes(), y_save.tobytes()))

    conn.commit()
    conn.close()
    print("  └─ 跨實驗格點 RMSE 數據已成功寫入資料庫！")


# ==========================================
# 4. 主執行程序
# ==========================================
if __name__ == "__main__":
    init_db(DB_FILE)
    """
    base_dir = "../plot/out2"
    experiments_patterns = {
        "SF2SF": os.path.join(base_dir, "SF2SF/netcdf/output_0_all.nc"),
        "SF2P":  os.path.join(base_dir, "SF2SFslp/netcdf/output_0_all.nc"),
        "P2P":   os.path.join(base_dir, "P2P25Y/netcdf/output_0_all.nc")
    }
    """
    base_dir = "output"
    experiments_patterns = {
        "P2P": os.path.join(base_dir, "P2P25Y/output_0_all.nc"),
        "CP2P":  os.path.join(base_dir, "CP2P25Y/output_0_all.nc"),
    }

    experiments_files = {}

    # 階段 1：計算各實驗時間軸上的 RMSE & FSS (Pred vs Truth)
    for exp_name, path_pattern in experiments_patterns.items():
        nc_files = sorted(glob.glob(path_pattern))
        experiments_files[exp_name] = nc_files
        print(f"\n[INFO] 開始處理實驗時間軸指標：{exp_name}，共找到 {len(nc_files)} 個檔案")

        for nc_file in nc_files:
            print(f"  └─ 處理檔案: {os.path.basename(nc_file)}")
            records = process_time_series_metrics(nc_file, exp_name=exp_name, var_name="precipitation")

            if records:
                save_time_metrics_to_db(records, db_path=DB_FILE)
                print(f"     已寫入 {len(records)} 筆 Pred vs Truth 指標數據")

    # 階段 2：預計算跨實驗格點 RMSE 散佈圖數據 (Pred vs Truth)
    precompute_grid_rmse_scatter(experiments_files, var_name="precipitation", max_points=50000, db_path=DB_FILE)

    print("\n[SUCCESS] 所有預計算流程順利完成！")
