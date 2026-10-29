# doubleml-pipeline（Google）でオレンジジュースのデータを分析するセル
# 前提：ノートブックの「1. データを読む」まで実行して、df（brand_name, log_price, log_comp_price を含む）があること
# インストール：%pip install doubleml-pipeline "flaml[automl]"
import pandas as pd
from doubleml_pipeline.preprocessing.scalers import CausalDataScaler
from doubleml_pipeline.workflow import CausalWorkflowOrchestrator

pdf = df.copy()

# パイプラインが必要とする列
# 週番号を日付に変換（週1を1989-09-14として、7日ずつ）
pdf["date"] = (pd.Timestamp("1989-09-14") + pd.to_timedelta((pdf["week"] - 1) * 7, unit="D")).dt.strftime("%Y-%m-%d")
pdf["geo"] = "store_" + pdf["store"].astype(str)   # 地域 = 店舗
pdf["item"] = pdf["brand_name"]                     # 商品 = ブランド

# ブランドはダミー列にして交絡に入れる
brand_dummies = pd.get_dummies(pdf["brand"], prefix="brand", drop_first=True).astype(int)
pdf = pd.concat([pdf, brand_dummies], axis=1)

# CATEの条件
pdf["flyer"] = (pdf["feat"] >= 0.5).astype(int)                 # チラシあり
pdf["large_pack"] = pdf["brand"].isin([2, 6, 11]).astype(int)   # 96oz・128oz

demo_cols = ["AGE60", "EDUC", "ETHNIC", "INCOME", "HHLARGE", "WORKWOM",
             "HVAL150", "SSTRDIST", "SSTRVOL", "CPDIST5", "CPWVOL5"]
x_cols_pl = ["store", "week", "deal", "feat", "log_comp_price"] + demo_cols + list(brand_dummies.columns)
pdf = pdf.reset_index(drop=True)

scaler = CausalDataScaler()   # log価格・log販売数・比率なので、スケーリングは不要

orchestrator = CausalWorkflowOrchestrator(
    y_col="logmove",
    d_cols=["log_price"],
    x_cols_list=[x_cols_pl],
    ml_models_y=["lgbm", "xgboost"],
    ml_models_t=["lgbm", "xgboost"],
    n_folds_list=[5],
    output_dir="../pipeline_results",
    time_budget=15,
    n_jobs=-1,          # メモリ不足で止まる場合は 1 にする
    date_col="date",
    geo_col="geo",
    item_col="item",
    covariates_for_cate_list=[["WORKWOM", "INCOME", "flyer", "large_pack"]],
    cate_structure_list=["auto_additive"],
    run_sensitivity=True,
    sensitivity_scope=["total"],
)

orchestrator.run_full_pipeline(pdf, scaler, top_n=2, n_reps=3)
# 結果：../pipeline_results/ の HTMLレポート（log_price_logmove.html）を開く
