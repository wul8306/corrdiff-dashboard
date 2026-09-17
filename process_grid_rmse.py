"""
預算跨實驗格點 RMSE 與比例 (Grid-level Pre-computation)
=====================================================
計算任意兩兩實驗組合 (SF2SF, SF2P, P2P) 的格點 RMSE 散佈圖數據，
並寫入 SQLite 資料庫，避免 Streamlit 重複讀取 NetCDF。
"""

import os
import sqlite3
import numpy as np
import xarray as xr
from itertools import combinations

DB_PATH = "corrdiff_rmse_metrics.db"

def init_grid_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 建立紀錄跨實驗格點 RMSE 統計資訊與點位資料的資料表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS grid_rmse_scatter (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exp_x TEXT NOT NULL,
        exp_y TEXT NOT NULL,
        var_name TEXT NOT NULL,
        ratio_y_gt_x REAL,
        x_values BLOB, -- 儲存抽樣後的 float32 陣列 (Numpy bytes)
        y_values BLOB, -- 儲存抽樣後的 float32 陣列 (Numpy bytes)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(exp_x, exp_y, var_name)
    );
    """)
    conn.commit()
    conn.close()

def get_grid_rmse(nc_path, var_name="precipitation"):
    ds_truth = xr.open_dataset(nc_path, group="truth")
    ds_pred = xr.open_dataset(nc_path, group="prediction")

    da_truth = ds_truth[var_name]
    
    if "ensemble" in ds_pred[var_name].dims:
        da_pred = ds_pred[var_name].mean(dim="ensemble")
    else:
        da_pred = ds_pred[var_name]

    if 'time' in da_truth.dims:
        grid_rmse = np.sqrt(((da_pred - da_truth) ** 2).mean(dim='time')).values.flatten()
    else:
        grid_rmse = np.sqrt((da_pred.values - da_truth.values) ** 2).flatten()

    ds_truth.close()
    ds_pred.close()
    return grid_rmse

def precompute_grid_rmse(experiments, var_name="precipitation", max_points=50000):
    init_grid_db(DB_PATH)
    
    # 1. 預先將所有實驗單獨的格點 RMSE 計算出來並暫存
    rmse_dict = {}
    for exp_name, nc_path in experiments.items():
        if os.path.exists(nc_path):
            print(f"[INFO] 正在計算 {exp_name} 的格點 RMSE...")
            rmse_dict[exp_name] = get_grid_rmse(nc_path, var_name)
        else:
            print(f"[WARNING] 找不到檔案: {nc_path}")

    exp_names = list(rmse_dict.keys())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 2. 計算任意兩兩實驗組合 (排列組合)
    for exp_x, exp_y in combinations(exp_names, 2):
        print(f"[PROCESS] 配對預算: {exp_x} (X) vs. {exp_y} (Y)")
        
        rmse_x = rmse_dict[exp_x]
        rmse_y = rmse_dict[exp_y]

        valid_mask = ~(np.isnan(rmse_x) | np.isnan(rmse_y))
        x_vals = rmse_x[valid_mask].astype(np.float32)
        y_vals = rmse_y[valid_mask].astype(np.float32)

        if len(x_vals) == 0:
            continue

        # 計算全域精準比例 Y > X
        ratio_y_gt_x = float((np.sum(y_vals > x_vals) / len(x_vals)) * 100.0)

        # 為了壓縮儲存空間與提升 Streamlit 載入速度，只抽樣 50,000 點存入 DB
        if len(x_vals) > max_points:
            idx = np.random.choice(len(x_vals), size=max_points, replace=False)
            x_save, y_save = x_vals[idx], y_vals[idx]
        else:
            x_save, y_save = x_vals, y_vals

        # 使用 tobytes() 轉為二进制 BLOB 高效寫入資料庫
        cursor.execute("""
        INSERT OR REPLACE INTO grid_rmse_scatter 
        (exp_x, exp_y, var_name, ratio_y_gt_x, x_values, y_values)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (exp_x, exp_y, var_name, ratio_y_gt_x, x_save.tobytes(), y_save.tobytes()))

    conn.commit()
    conn.close()
    print("[SUCCESS] 格點 RMSE 預算並寫入資料庫完成！")

if __name__ == "__main__":
    base_dir = "/syn_sata8/users/lct/corrdiff/plot/out2"
    experiments = {
        "SF2SF": os.path.join(base_dir, "SF2SF/netcdf/output_0_all.nc"),
        "SF2P":  os.path.join(base_dir, "SF2SFslp/netcdf/output_0_all.nc"),
        "P2P":   os.path.join(base_dir, "P2P25Y/netcdf/output_0_all.nc")
    }
    
    precompute_grid_rmse(experiments, var_name="precipitation")
