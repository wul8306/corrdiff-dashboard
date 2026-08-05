"""
NetCDF Groups (Truth / Prediction / Input) 指標預計算腳本
======================================================
計算:
1. Prediction (Ensemble Mean) vs. Truth
2. Input vs. Truth
並將 RMSE 與 FSS 存入 SQLite，供 GitHub / Streamlit 網頁展示。
"""

import os
import glob
import sqlite3
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

def init_db(db_path="corrdiff_metrics.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiment_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exp_name TEXT NOT NULL,
        file_name TEXT NOT NULL,
        valid_time TEXT,
        comparison_type TEXT, -- 'pred_vs_truth' 或 'input_vs_truth'
        variable_name TEXT,   -- 'precipitation', 'temperature_2m', etc.
        rmse REAL,
        fss_threshold REAL,
        fss_window_size INTEGER,
        fss_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(exp_name, file_name, valid_time, comparison_type, variable_name, fss_threshold, fss_window_size)
    );
    """)
    conn.commit()
    conn.close()

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

def process_grouped_nc(nc_path, exp_name, var_name="precipitation"):
    results = []
    
    try:
        # 透過 group 參數分別讀取 truth, input, prediction
        ds_truth = xr.open_dataset(nc_path, group="truth")
        ds_input = xr.open_dataset(nc_path, group="input")
        ds_pred = xr.open_dataset(nc_path, group="prediction")
    except Exception as e:
        print(f"[ERROR] 讀取 Group 失敗 {nc_path}: {e}")
        return results

    if var_name not in ds_truth:
        print(f"[WARNING] 檔案內找不到變數 {var_name}")
        return results

    # 1. 處理 Prediction: 對 ensemble 取平均 (Ensemble Mean)
    da_truth = ds_truth[var_name]
    da_input = ds_input[var_name]
    da_pred_mean = ds_pred[var_name].mean(dim="ensemble") # 沿著 ensemble 維度取平均

    fss_thresholds = [0.1, 1.0, 5.0, 10.0, 20.0, 35.0]
    fss_windows = [3, 5, 9, 15, 25]

    times = da_truth['time'].values if 'time' in da_truth.coords else [None]

    for t_idx, t_val in enumerate(times):
        time_str = str(t_val) if t_val is not None else "N/A"
        
        # 擷取單一時間步 2D 矩陣
        truth_2d = da_truth.isel(time=t_idx).values if 'time' in da_truth.coords else da_truth.values
        input_2d = da_input.isel(time=t_idx).values if 'time' in da_input.coords else da_input.values
        pred_2d = da_pred_mean.isel(time=t_idx).values if 'time' in da_pred_mean.coords else da_pred_mean.values

        # 對兩種對比分別計算： (1) Prediction vs. Truth  (2) Input vs. Truth
        comparisons = {
            "pred_vs_truth": pred_2d,
            "input_vs_truth": input_2d
        }

        for comp_type, target_2d in comparisons.items():
            valid_mask = ~(np.isnan(truth_2d) | np.isnan(target_2d))
            if not np.any(valid_mask):
                continue

            # 計算 RMSE
            rmse = float(np.sqrt(np.mean((target_2d[valid_mask] - truth_2d[valid_mask]) ** 2)))

            # 計算 FSS (僅針對降雨 precipitation，或適當指標)
            truth_filled = np.nan_to_num(truth_2d, nan=0.0)
            target_filled = np.nan_to_num(target_2d, nan=0.0)

            for th in fss_thresholds:
                for w in fss_windows:
                    fss_val = calculate_fss(truth_filled, target_filled, th, w)
                    results.append((
                        exp_name,
                        os.path.basename(nc_path),
                        time_str,
                        comp_type,
                        var_name,
                        rmse,
                        th,
                        w,
                        fss_val
                    ))

    ds_truth.close()
    ds_input.close()
    ds_pred.close()
    return results

def batch_save_to_db(results, db_path="corrdiff_metrics.db"):
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

if __name__ == "__main__":
    init_db("corrdiff_metrics.db")
    # 執行單一檔案範例
    records = process_grouped_nc("/syn_sata8/users/lct/corrdiff/plot/out2/SF2SF/netcdf/output_0_all.nc", exp_name="Exp_v1")
    batch_save_to_db(records)
