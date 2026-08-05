# 🌧️ CorrDiff 氣象降尺度實驗評估系統 (Evaluation Dashboard)

本專案專為氣象 AI 生成式降尺度模型（NVIDIA CorrDiff）設計，提供自動化的評估指標計算與互動式 Web Dashboard 介面。能夠自動讀取 NetCDF 檔案中的 `truth`、`prediction`（含系集成員維度 `ensemble`）與 `input` 群組，極速預計算 **RMSE** 與 **FSS (Fractions Skill Score)**，並透過輕量化的 SQLite 資料庫進行互動式視覺化呈現。

---

## 🌟 核心功能與特色

* **支持 NetCDF4 Group 結構**：自動解析包含 `truth`、`prediction` 及 `input` 的複雜 NetCDF4 檔案。
* **高效 FSS 計算**：採用 2D 均值濾波（Uniform Filter / 卷積演算法），極速計算多種降雨門檻與空間滑動視窗下的 FSS 分數。
* **離線與分離式架構**：將大型 `.nc` 檔案的計算結果濃縮寫入 SQLite 資料庫 (`corrdiff_metrics.db`)，前端網頁只需讀取幾百 KB 的資料庫即可載入圖表。
* **動態互動 Dashboard**：支援透過介面拉條（Slider）與下拉選單動態調整：
  * 降雨門檻 (Threshold, e.g., 1.0, 10.0, 20.0 mm/hr)
  * 空間視窗大小 (Neighborhood Window, e.g., 3x3, 5x5, 15x15)
  * 比較類型 (`Prediction vs. Truth` 與 `Input vs. Truth`)

---

## 📂 專案架構

```text
corrdiff-dashboard/
├── process_corrdiff_nc.py  # 核心腳本：讀取 NetCDF 群組，預計算 RMSE/FSS 並寫入 SQLite
├── app.py                  # 前端網頁：Streamlit 互動 Dashboard
├── corrdiff_metrics.db     # 預計算指標資料庫 (可上傳至 GitHub 作為 Demo)
├── requirements.txt        # 套件依賴清單
├── .gitignore              # 忽略原始大型 .nc 檔案
└── README.md               # 專案說明文件
