import json
import os
import sqlite3
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
import yaml

# ==============================================================================
#   載入 YAML 設定檔 for experiments
# ==============================================================================
def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 找不到設定檔: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

# ==============================================================================
# 一、核心計算模組 (Core Computation Module)
# ==============================================================================

def load_land_sea_mask(mask_path: str, mask_var: str = "land_sea_mask") -> xr.DataArray:
    """
    載入 Land-Sea Mask 檔案
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"❌ 找不到 Land-Sea Mask 檔案: {mask_path}")
    
    ds = xr.open_dataset(mask_path)
    if mask_var not in ds:
        raise KeyError(f"❌ Mask 檔案中找不到變數 [{mask_var}]，可用的變數為: {list(ds.keys())}")

    return ds[mask_var].load()


def calculate_fss(obs_binary: np.ndarray, pred_binary: np.ndarray, window_size: int) -> float:
    """
    使用 2D 滑動視窗 (Uniform Filter) 高效計算 Fractional Skill Score (FSS)
    """
    obs_frac = uniform_filter(obs_binary, size=window_size, mode='constant', cval=0.0)
    pred_frac = uniform_filter(pred_binary, size=window_size, mode='constant', cval=0.0)

    mse = np.mean((obs_frac - pred_frac) ** 2)
    mse_ref = np.mean(obs_frac ** 2 + pred_frac ** 2)

    if mse_ref == 0:
        return 1.0 if mse == 0 else 0.0

    fss = 1.0 - (mse / mse_ref)
    return float(fss)


def calculate_grid_rmse(da_truth: xr.DataArray, da_pred_mean: xr.DataArray) -> np.ndarray:
    """
    計算空間中每一個獨立網格點 (Grid-level) 的平均 RMSE 陣列 (展平為 1D)
    """
    sq_err = (da_pred_mean - da_truth) ** 2

    if 'time' in da_truth.dims and da_truth.sizes['time'] > 0:
        if np.issubdtype(da_truth['time'].dtype, np.datetime64):
            try:
                daily_rmse = np.sqrt(sq_err.resample(time='1D').mean(dim='time', skipna=True))
                mean_rmse = daily_rmse.mean(dim='time', skipna=True)
            except Exception:
                mean_rmse = np.sqrt(sq_err.mean(dim='time', skipna=True))
        else:
            mean_rmse = np.sqrt(sq_err.mean(dim='time', skipna=True))

        grid_rmse = mean_rmse.values.flatten().astype(np.float32)
    else:
        grid_rmse = np.sqrt(sq_err.values).flatten().astype(np.float32)

    return grid_rmse


def calculate_rmse_t_series(da_truth: xr.DataArray, da_pred_mean: xr.DataArray) -> tuple[np.ndarray, list[str]]:
    """
    計算每個時間步長 (Time Step) 的全空間區域 RMSE (rmse_t) 與時間標籤
    """
    if 'time' not in da_truth.dims:
        truth_2d = da_truth.values
        pred_2d = da_pred_mean.values
        valid_mask = ~(np.isnan(truth_2d) | np.isnan(pred_2d))
        rmse_val = float(np.sqrt(np.mean((pred_2d[valid_mask] - truth_2d[valid_mask]) ** 2))) if np.any(valid_mask) else np.nan
        return np.array([rmse_val], dtype=np.float32), ["STATIC"]

    raw_times = da_truth['time'].values
    if np.issubdtype(raw_times.dtype, np.datetime64):
        time_series = pd.to_datetime(raw_times)
        times_str = [t.strftime('%Y-%m-%d %H:%M') for t in time_series]
    else:
        base_date = pd.Timestamp("2016-01-01 00:00:00")
        times_str = [(base_date + pd.Timedelta(days=float(t))).strftime('%Y-%m-%d %H:%M') for t in raw_times]

    rmse_t_list = []
    n_times = len(times_str)

    for t_idx in range(n_times):
        truth_2d = da_truth.isel(time=t_idx).values
        pred_2d = da_pred_mean.isel(time=t_idx).values

        valid_mask = ~(np.isnan(truth_2d) | np.isnan(pred_2d))
        if np.any(valid_mask):
            sq_diff = (pred_2d[valid_mask] - truth_2d[valid_mask]) ** 2
            rmse_val = float(np.sqrt(np.mean(sq_diff)))
        else:
            rmse_val = np.nan

        rmse_t_list.append(rmse_val)

    return np.array(rmse_t_list, dtype=np.float32), times_str


def calculate_rmse_t_season(rmse_t_arr: np.ndarray, times_str: list[str]) -> dict:
    """
    計算特定五個季節 (DJF, MA, MJ, JAS, ON) 的平均 RMSE。
    """
    seasons = ['DJF', 'MA', 'MJ', 'JAS', 'ON']

    if not times_str or times_str[0] == "STATIC":
        return {s: np.nan for s in seasons}

    df = pd.DataFrame({
        'rmse': rmse_t_arr,
        'time': pd.to_datetime(times_str)
    })

    season_map = {
        12: 'DJF', 1: 'DJF', 2: 'DJF',
        3: 'MA', 4: 'MA',
        5: 'MJ', 6: 'MJ',
        7: 'JAS', 8: 'JAS', 9: 'JAS',
        10: 'ON', 11: 'ON'
    }

    df['season'] = df['time'].dt.month.map(season_map)
    seasonal_means = df.groupby('season')['rmse'].mean().to_dict()

    return {s: float(seasonal_means.get(s, np.nan)) for s in seasons}


# ==============================================================================
# 二、綜合指標計算與評估 (Full Metrics Evaluation)
# ==============================================================================

def compute_experiment_metrics(nc_path: str, var_name: str = "precipitation",
                               fss_thresholds: list = None, fss_windows: list = None,
                               mask_da: xr.DataArray = None, mask_value: float = 1.0) -> dict:
    """
    對單一實驗 NetCDF 進行完整指標計算 (支援 Land-Sea Mask 遮罩過濾)
    """
    if fss_thresholds is None:
        fss_thresholds = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0]
    if fss_windows is None:
        fss_windows = [3, 5, 7, 9, 15, 20]

    try:
        with xr.open_dataset(nc_path, group="truth") as ds_truth, xr.open_dataset(nc_path, group="prediction") as ds_pred:
            da_truth = ds_truth[var_name].load()
            da_pred = ds_pred[var_name].load()
    except Exception:
        with xr.open_dataset(nc_path) as ds:
            da_truth = ds[var_name].load()
            da_pred = ds[var_name].load()

    if "ensemble" in da_pred.dims:
        da_pred = da_pred.mean(dim="ensemble")

    # 🔹 套用 Land-Sea Mask 遮罩 (非目標區域設為 NaN)
    if mask_da is not None:
        mask_cond = (mask_da.values == mask_value)
        da_truth = da_truth.where(mask_cond)
        da_pred = da_pred.where(mask_cond)

    # 1. 格點級 RMSE (Grid-level)
    grid_rmse_arr = calculate_grid_rmse(da_truth, da_pred)

    # 2. 時間軸 RMSE (Time-level)
    rmse_t_arr, times_str = calculate_rmse_t_series(da_truth, da_pred)

    # 3. 季節性時間軸 RMSE (Seasonal-level)
    seasonal_metrics = calculate_rmse_t_season(rmse_t_arr, times_str)

    # 4. 多尺度 FSS 計算
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
            obs_binary = (truth_filled >= th).astype(np.float32)
            pred_binary = (pred_filled >= th).astype(np.float32)
            for w in fss_windows:
                fss_val = calculate_fss(obs_binary, pred_binary, w)
                time_fss_dict[(th, w)].append(fss_val)

    avg_fss_dict = {key: float(np.mean(val)) if val else 0.0 for key, val in time_fss_dict.items()}

    return {
        "grid_rmse": grid_rmse_arr,
        "rmse_t": rmse_t_arr,
        "times_str": times_str,
        "avg_fss": avg_fss_dict,
        "seasonal_metrics": seasonal_metrics
    }


# ==============================================================================
# 三、預計算與 SQLite 資料庫導出模組 (Precomputation & Database Export)
# ==============================================================================

def init_database(db_path: str):
    """建立 SQLite 資料庫及對應資料表"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS grid_rmse_scatter")
    cursor.execute("DROP TABLE IF EXISTS time_rmse_scatter")
    cursor.execute("DROP TABLE IF EXISTS summary_metrics")
    cursor.execute("DROP TABLE IF EXISTS seasonal_rmse")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS grid_rmse_scatter (
            exp_x TEXT, exp_y TEXT, var_name TEXT, valid_time TEXT, ratio_y_gt_x REAL, x_values BLOB, y_values BLOB,
            PRIMARY KEY (exp_x, exp_y, var_name)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS time_rmse_scatter (
            exp_x TEXT, exp_y TEXT, var_name TEXT, valid_time TEXT, ratio_y_gt_x REAL, x_values BLOB, y_values BLOB,
            PRIMARY KEY (exp_x, exp_y, var_name)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summary_metrics (
            exp_name TEXT, file_name TEXT, time_label TEXT, comparison_type TEXT, variable_name TEXT,
            rmse REAL, fss_threshold REAL, fss_window_size INTEGER, fss_score REAL,
            PRIMARY KEY (exp_name, file_name, variable_name, fss_threshold, fss_window_size)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seasonal_rmse (
            exp_name TEXT,
            season TEXT,
            variable_name TEXT,
            rmse_mean REAL,
            PRIMARY KEY (exp_name, season, variable_name)
        )
    """)

    conn.commit()
    conn.close()


def run_precomputations(exp_nc_map: dict[str, str], db_path: str = "corrdiff_allmetrics_masked.db", 
                        var_name: str = "precipitation",
                        mask_path: str = None, mask_var: str = "land_sea_mask", mask_value: float = 1.0):
    """
    執行完整預計算，套用 Land-Sea Mask 遮罩並寫入指定的 SQLite 資料庫
    
    :param mask_path: Land-Sea Mask NetCDF 檔案路徑 (若為 None 則不使用遮罩)
    :param mask_var: Mask 變數名稱 (例如 'land_sea_mask' 或 'lsm')
    :param mask_value: 被選取的目標數值 (例如 1.0 代表陸地，0.0 代表海洋)
    """
    init_database(db_path)

    # 載入 Mask (若有指定)
    mask_da = None
    if mask_path:
        print(f"🗺️ 載入 Land-Sea Mask 遮罩: {mask_path} (變數: {mask_var}, 選取值: {mask_value})")
        mask_da = load_land_sea_mask(mask_path, mask_var=mask_var)

    exp_results = {}
    for exp_name, nc_path in exp_nc_map.items():
        if os.path.exists(nc_path):
            print(f"⏳ 正在分析實驗與計算指標: [{exp_name}] -> {nc_path}")
            exp_results[exp_name] = compute_experiment_metrics(
                nc_path, var_name=var_name, mask_da=mask_da, mask_value=mask_value
            )
        else:
            print(f"⚠️ 找不到檔案，跳過: {nc_path}")

    exp_names = list(exp_results.keys())
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        print("\n📝 正在寫入 Summary Metrics (FSS & Overall RMSE)...")
        for exp_name, data in exp_results.items():
            avg_rmse = float(np.nanmean(data['rmse_t']))
            nc_filename = os.path.basename(exp_nc_map[exp_name])

            for (th, w), avg_fss in data['avg_fss'].items():
                cursor.execute("""
                    INSERT OR REPLACE INTO summary_metrics
                    (exp_name, file_name, time_label, comparison_type, variable_name, rmse, fss_threshold, fss_window_size, fss_score)
                    VALUES (?, ?, 'ALL_TIME_AVG', 'pred_vs_truth', ?, ?, ?, ?, ?)
                """, (exp_name, nc_filename, var_name, avg_rmse, th, w, avg_fss))

            # 將季節性 RMSE 寫入 seasonal_rmse 資料表
            for season, s_rmse in data['seasonal_metrics'].items():
                if not np.isnan(s_rmse):
                    cursor.execute("""
                        INSERT OR REPLACE INTO seasonal_rmse
                        (exp_name, season, variable_name, rmse_mean)
                        VALUES (?, ?, ?, ?)
                    """, (exp_name, season, var_name, s_rmse))

        print("\n🔗 正在計算兩兩配對比對 (Pairwise Comparisons)...")
        for i in range(len(exp_names)):
            for j in range(i + 1, len(exp_names)):
                exp_x, exp_y = exp_names[i], exp_names[j]

                # A. 格點級散佈圖 (已排除 Mask 掉的 NaN 點)
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
                    print(f"  ├─ [Grid Scatter] {exp_x} vs {exp_y}: {len(gx_v)} 個有效網格點, Y > X 比例: {g_ratio:.2f}%")

                # B. 時間軸級散佈圖 (基於 Mask 區域計算的 rmse_t)
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

    print(f"\n🎉 遮罩計算完成！所有指標與散佈圖資料已更新至新資料庫: [{db_path}]")


# ==============================================================================
# 四、主程式執行進入點
# ==============================================================================

if __name__ == "__main__":
    # 1. 載入 YAML 設定檔
    CONFIG_PATH = "config.yaml"
    config = load_config(CONFIG_PATH)

    # 2. 提取實驗映射圖與相關設定
    EXPERIMENT_NC_MAP = config["experiments"]
    VAR_NAME = config["dataset"]["variable_name"]
    DB_PATH = config["dataset"]["db_path"]

    # 3. 設定 Land-Sea Mask 檔案與參數
    MASK_PATH = config["dataset"]["mask"]["path"]
    MASK_VAR = config["dataset"]["mask"]["variable_name"]
    MASK_VAL = config["dataset"]["mask"]["target_value"]

    print(f"✅ 成功載入 {len(EXPERIMENT_NC_MAP)} 個實驗設定")
    for exp_name, nc_path in EXPERIMENT_NC_MAP.items():
        print(f"  • {exp_name} -> {nc_path}")

    # 4 指定新的 SQLite 資料庫檔名
    DB_PATH = "corrdiff_allmetrics_sea_land.db"
    print(f"\n ✅ 輸出db設定:  {DB_PATH}")


    # 5. 執行預計算與資料庫寫入
    if os.path.exists(MASK_PATH):
        run_precomputations(
            exp_nc_map=EXPERIMENT_NC_MAP,
            db_path=DB_PATH,
            var_name=VAR_NAME,
            mask_path=MASK_PATH,
            mask_var=MASK_VAR,
            mask_value=MASK_VAL
        )
    else:
        print(f"⚠️ 找不到 Mask 檔案 ({MASK_PATH})，請先確認 config.yaml 中的路徑設定。")

