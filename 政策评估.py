"""
模块四：政策效果自动评估流水线
功能：将你的因果推断技能（PSM-DID/断点回归）与LLM结合
实现：输入政策名称 → 自动完成数据处理+计量分析+LLM解读+报告生成

安装依赖：
  pip install pandas numpy statsmodels linearmodels econml matplotlib seaborn
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
matplotlib.rcParams['axes.unicode_minus'] = False
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


# ========== 数据准备模块 ==========
class PolicyDataPreparer:
    """
    准备政策评估所需面板数据
    数据来源：Wind/国泰安/同花顺等数据库导出的CSV
    """
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path, encoding="utf-8-sig")
        logger.info(f"数据加载成功：{self.df.shape[0]}行 × {self.df.shape[1]}列")

    def prepare_did_data(self, policy_year: int, treat_col: str,
                         outcome_col: str, control_cols: List[str],
                         entity_col="企业代码", time_col="年份") -> pd.DataFrame:
        """
        准备DID（双重差分）分析数据
        Args:
            policy_year: 政策实施年份
            treat_col: 处理组标识列（1=受政策影响，0=对照组）
            outcome_col: 结果变量（如：研发投入/营业收入/ESG评分）
            control_cols: 控制变量列表
        """
        df = self.df.copy()
        # 生成时间虚拟变量（政策实施后=1）
        df["post"] = (df[time_col] >= policy_year).astype(int)
        # 核心DID交互项
        df["did"] = df[treat_col] * df["post"]
        # 对数化处理（减少异方差）
        for col in [outcome_col] + control_cols:
            if df[col].min() > 0:
                df[f"ln_{col}"] = np.log(df[col])
        return df

    def prepare_rdd_data(self, cutoff: float, running_var: str,
                         outcome_col: str, bandwidth: float = None) -> pd.DataFrame:
        """
        准备断点回归（RDD）数据
        Args:
            cutoff: 政策门槛值（如：企业规模达到某阈值获得政策支持）
            running_var: 驱动变量（连续变量，如企业年营收）
            bandwidth: 带宽（若为None则自动计算最优带宽）
        """
        df = self.df.copy()
        df["running_centered"] = df[running_var] - cutoff  # 中心化
        df["treated"] = (df[running_var] >= cutoff).astype(int)
        if bandwidth:
            df = df[df["running_centered"].abs() <= bandwidth]
        return df


# ========== 计量分析模块 ==========
class CausalInferenceAnalyzer:
    """实现PSM-DID、断点回归等因果推断方法"""

    # ---------- PSM倾向得分匹配 ----------
    @staticmethod
    def psm_matching(df: pd.DataFrame, treat_col: str,
                     match_vars: List[str], caliper=0.05) -> pd.DataFrame:
        """
        倾向得分匹配（PSM）
        使用Logit模型估计倾向得分，最近邻匹配
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = df[match_vars].fillna(df[match_vars].median())
        y = df[treat_col]

        # 估计倾向得分
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logit = LogisticRegression(max_iter=1000)
        logit.fit(X_scaled, y)
        df = df.copy()
        df["pscore"] = logit.predict_proba(X_scaled)[:, 1]

        # 最近邻匹配（带caliper）
        treat_df = df[df[treat_col] == 1].copy()
        ctrl_df = df[df[treat_col] == 0].copy()
        matched_ctrl_ids = []

        for _, t_row in treat_df.iterrows():
            diffs = (ctrl_df["pscore"] - t_row["pscore"]).abs()
            min_diff = diffs.min()
            if min_diff <= caliper:
                matched_ctrl_ids.append(diffs.idxmin())

        matched_ctrl = ctrl_df.loc[matched_ctrl_ids]
        matched_df = pd.concat([treat_df, matched_ctrl])
        logger.info(f"PSM匹配完成：处理组{len(treat_df)}家，匹配对照组{len(matched_ctrl)}家")
        return matched_df

    # ---------- DID回归 ----------
    @staticmethod
    def did_regression(df: pd.DataFrame, outcome: str,
                       did_col="did", treat_col="treat",
                       post_col="post", controls: List[str] = None,
                       entity_col="企业代码", time_col="年份") -> Dict:
        """
        双重差分回归（含双向固定效应）
        """
        try:
            from linearmodels.panel import PanelOLS
            # 设置面板数据索引
            df_panel = df.set_index([entity_col, time_col])
            ctrl_str = " + ".join(controls) if controls else ""
            formula = f"{outcome} ~ {did_col} + {ctrl_str} + EntityEffects + TimeEffects"
            model = PanelOLS.from_formula(formula, data=df_panel, drop_absorbed=True)
            res = model.fit(cov_type="clustered", cluster_entity=True)
        except Exception:
            # 降级为OLS
            ctrl_str = " + ".join(controls) if controls else ""
            formula = f"{outcome} ~ {did_col} + {treat_col} + {post_col}" + (f" + {ctrl_str}" if ctrl_str else "")
            res = ols(formula, data=df).fit(cov_type="HC1")

        did_coef = res.params.get(did_col, res.params.get(f"{did_col}", np.nan))
        did_pval = res.pvalues.get(did_col, np.nan)
        return {
            "did_coefficient": round(float(did_coef), 4),
            "p_value": round(float(did_pval), 4),
            "significant": did_pval < 0.05,
            "r_squared": round(float(res.rsquared), 4),
            "n_obs": int(res.nobs),
            "model_summary": str(res.summary),
        }

    # ---------- 断点回归 ----------
    @staticmethod
    def rdd_regression(df: pd.DataFrame, outcome: str,
                       running="running_centered", treated="treated",
                       poly_order=1) -> Dict:
        """
        断点回归（支持多项式阶数）
        """
        df = df.copy()
        # 构建多项式项
        for p in range(1, poly_order + 1):
            df[f"run_{p}"] = df[running] ** p
            df[f"run_{p}_treat"] = df[f"run_{p}"] * df[treated]

        poly_terms = " + ".join([f"run_{p} + run_{p}_treat" for p in range(1, poly_order + 1)])
        formula = f"{outcome} ~ {treated} + {poly_terms}"
        res = ols(formula, data=df).fit(cov_type="HC1")

        rdd_coef = res.params.get(treated, np.nan)
        rdd_pval = res.pvalues.get(treated, np.nan)
        return {
            "rdd_coefficient": round(float(rdd_coef), 4),
            "p_value": round(float(rdd_pval), 4),
            "significant": rdd_pval < 0.05,
            "r_squared": round(float(res.rsquared), 4),
            "n_obs": int(res.nobs),
        }

    # ---------- 平行趋势检验 ----------
    @staticmethod
    def parallel_trend_test(df: pd.DataFrame, outcome: str,
                            treat_col: str, base_year: int,
                            time_col="年份") -> Dict:
        """
        DID前提检验：处理组和对照组在政策前是否有相似趋势
        """
        years = sorted(df[time_col].unique())
        coefs, ci_low, ci_high, yr_labels = [], [], [], []

        for yr in years:
            if yr == base_year:
                continue
            df_temp = df.copy()
            df_temp["yr_dummy"] = (df_temp[time_col] == yr).astype(int)
            df_temp["interaction"] = df_temp[treat_col] * df_temp["yr_dummy"]
            res = ols(f"{outcome} ~ interaction + {treat_col} + yr_dummy", data=df_temp).fit()
            coef = res.params.get("interaction", 0)
            ci = res.conf_int().loc["interaction"] if "interaction" in res.conf_int().index else [0, 0]
            coefs.append(coef)
            ci_low.append(ci[0])
            ci_high.append(ci[1])
            yr_labels.append(str(yr))

        return {"years": yr_labels, "coefficients": coefs, "ci_low": ci_low, "ci_high": ci_high}


# ========== LLM解读模块 ==========
class PolicyEvalInterpreter:
    """
    将计量分析结果转化为政策语言报告
    核心价值：让非统计背景的领导和客户能看懂数据
    """
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def interpret_did(self, did_result: Dict, policy_name: str,
                      outcome_name: str) -> str:
        prompt = f"""
你是专业的政策评估研究员，请将以下双重差分（DID）分析结果，
用政府研究报告的语言风格进行专业解读，要求：
1. 解释政策效果的方向和大小
2. 说明结果的统计显著性
3. 给出政策效果的政策含义
4. 提出政策优化建议

政策名称：{policy_name}
结果变量：{outcome_name}
DID系数：{did_result['did_coefficient']}
P值：{did_result['p_value']}（{'显著' if did_result['significant'] else '不显著'}）
样本量：{did_result['n_obs']}
R²：{did_result['r_squared']}

请输出300-500字的政策解读。
"""
        return self.llm.generate(
            "你是专业政策评估研究员，擅长将统计结果转化为政策语言。",
            prompt, max_tokens=800, temperature=0.4
        )


# ========== 全流程整合 ==========
class PolicyEvaluationPipeline:
    """
    一键完成：数据准备 → 因果分析 → LLM解读 → 报告生成
    """
    def __init__(self, data_path: str, llm_engine=None):
        self.preparer = PolicyDataPreparer(data_path)
        self.analyzer = CausalInferenceAnalyzer()
        self.interpreter = PolicyEvalInterpreter(llm_engine) if llm_engine else None

    def run_did_evaluation(self, policy_name: str, policy_year: int,
                           treat_col: str, outcome_col: str,
                           control_cols: List[str], match_vars: List[str]) -> Dict:
        """完整的PSM-DID评估流程"""
        logger.info(f"开始评估政策：{policy_name}")

        # Step 1: 数据准备
        df = self.preparer.prepare_did_data(policy_year, treat_col,
                                             outcome_col, control_cols)
        # Step 2: PSM匹配
        df_matched = self.analyzer.psm_matching(df, treat_col, match_vars)

        # Step 3: DID回归
        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df_matched.columns else outcome_col
        did_res = self.analyzer.did_regression(
            df_matched, outcome_ln,
            controls=[f"ln_{c}" if f"ln_{c}" in df_matched.columns else c for c in control_cols]
        )

        # Step 4: 平行趋势检验
        pt_res = self.analyzer.parallel_trend_test(df_matched, outcome_ln, treat_col, policy_year - 1)

        # Step 5: LLM解读
        interpretation = ""
        if self.interpreter:
            interpretation = self.interpreter.interpret_did(did_res, policy_name, outcome_col)

        result = {
            "policy_name": policy_name,
            "did_results": did_res,
            "parallel_trend": pt_res,
            "llm_interpretation": interpretation,
        }

        # Step 6: 保存结果
        with open(f"eval_{policy_name[:10]}.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in result.items() if k != "parallel_trend"}, f,
                      ensure_ascii=False, indent=2)
        logger.info("评估完成，结果已保存")
        return result


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 模拟数据（实际使用Wind/国泰安数据）
    np.random.seed(42)
    n = 500
    mock_data = pd.DataFrame({
        "企业代码": np.repeat(range(100), 5),
        "年份": np.tile([2018, 2019, 2020, 2021, 2022], 100),
        "treat": np.repeat(np.random.binomial(1, 0.5, 100), 5),
        "研发投入": np.random.lognormal(10, 1, n),
        "营业收入": np.random.lognormal(12, 1.5, n),
        "资产总额": np.random.lognormal(11, 1.2, n),
        "员工人数": np.random.lognormal(6, 0.8, n),
    })
    # 模拟政策效应（处理组政策后研发投入增加15%）
    mask = (mock_data["treat"] == 1) & (mock_data["年份"] >= 2020)
    mock_data.loc[mask, "研发投入"] *= 1.15
    mock_data.to_csv("mock_panel_data.csv", index=False, encoding="utf-8-sig")

    # 运行评估（不含LLM解读）
    pipeline = PolicyEvaluationPipeline("mock_panel_data.csv", llm_engine=None)
    result = pipeline.run_did_evaluation(
        policy_name="专精特新企业认定政策",
        policy_year=2020,
        treat_col="treat",
        outcome_col="研发投入",
        control_cols=["营业收入", "资产总额", "员工人数"],
        match_vars=["营业收入", "资产总额", "员工人数"],
    )

    print(f"\n{'='*50}")
    print(f"政策：{result['policy_name']}")
    print(f"DID系数：{result['did_results']['did_coefficient']}")
    print(f"P值：{result['did_results']['p_value']}")
    print(f"是否显著：{result['did_results']['significant']}")
    if result["llm_interpretation"]:
        print(f"\nLLM政策解读：\n{result['llm_interpretation']}")