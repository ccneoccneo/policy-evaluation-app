"""
模块四：政策效果自动评估流水线（集成通义千问 LLM）
功能：PSM-DID、断点回归、双重机器学习，自动平行趋势检验 + 通义千问报告生成
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from typing import Dict, List, Optional, Tuple
import json
import logging
import sys
import dashscope
from http import HTTPStatus

logger = logging.getLogger(__name__)


# ========== 通义千问 LLM 引擎（基于您提供的 safe_api_call） ==========
class QwenEngine:
    """使用 dashscope SDK 调用通义千问"""
    def __init__(self, api_key: str = None, model: str = "qwen-turbo"):
        """
        Args:
            api_key: 阿里云 DashScope API Key，若不提供则从环境变量 DASHSCOPE_API_KEY 读取
            model: 模型名称，可选 qwen-turbo, qwen-plus, qwen-max
        """
        DASHSCOPE_API_KEY= "sk-2f95ab436b644f11849c067a74744c7a"
        if api_key:
            self.api_key = api_key
        else:
            import os
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
            if not self.api_key:
                raise ValueError("请提供 API Key 或设置环境变量 DASHSCOPE_API_KEY")
        self.model = model
        dashscope.api_key = self.api_key

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 800, temperature: float = 0.4) -> str:
        """调用通义千问生成回复，完全兼容 PolicyEvalInterpreter 的接口"""
        # 构造完整 prompt（将 system 和 user 合并，因为您的 safe_api_call 只接受一个 prompt）
        full_prompt = f"系统指令：{system_prompt}\n\n用户问题：{user_prompt}"
        return self._safe_api_call(full_prompt, max_tokens, temperature)

    def _safe_api_call(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """您提供的 safe_api_call 函数，稍作调整以接收参数"""
        try:
            print(f"调用通义千问 API，提示长度: {len(prompt)}", file=sys.stderr)

            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的政策评估研究员，擅长将计量经济学结果转化为政府研究报告风格的语言。请用中文回答。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8
            )

            print(f"API 状态码: {response.status_code}", file=sys.stderr)

            if response.status_code != HTTPStatus.OK:
                error_msg = f"API 调用失败: [{response.code}] {response.message}"
                print(f"错误: {error_msg}", file=sys.stderr)
                return f"抱歉，AI助手暂时无法回答。错误: {response.message}"

            # 提取响应内容
            if hasattr(response, 'output') and response.output:
                if hasattr(response.output, 'choices') and response.output.choices:
                    if len(response.output.choices) > 0:
                        choice = response.output.choices[0]
                        if hasattr(choice, 'message') and choice.message:
                            if hasattr(choice.message, 'content'):
                                content = choice.message.content
                                print(f"成功获取响应，长度: {len(content)}", file=sys.stderr)
                                return content

            # 备用提取方式
            if hasattr(response, 'output') and hasattr(response.output, 'text'):
                return response.output.text

            return "根据计量分析结果，未能生成有效的政策解读。"

        except Exception as e:
            print(f"API 调用异常: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return f"系统错误: {str(e)}"


# ========== 数据准备模块 ==========
class PolicyDataPreparer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path, encoding="utf-8-sig")
        logger.info(f"数据加载成功：{self.df.shape[0]}行 × {self.df.shape[1]}列")

    def prepare_did_data(self, policy_year: int, treat_col: str,
                         outcome_col: str, control_cols: List[str],
                         entity_col="企业代码", time_col="年份") -> pd.DataFrame:
        df = self.df.copy()
        df["post"] = (df[time_col] >= policy_year).astype(int)
        df["did"] = df[treat_col] * df["post"]
        for col in [outcome_col] + control_cols:
            if df[col].min() > 0:
                df[f"ln_{col}"] = np.log(df[col])
        return df

    def prepare_rdd_data(self, cutoff: float, running_var: str,
                         outcome_col: str, bandwidth: float = None) -> pd.DataFrame:
        df = self.df.copy()
        df["running_centered"] = df[running_var] - cutoff
        df["treated"] = (df[running_var] >= cutoff).astype(int)
        if bandwidth:
            df = df[df["running_centered"].abs() <= bandwidth]
        return df

    def prepare_dml_data(self, outcome_col: str, treat_col: str,
                         feature_cols: List[str], time_col: str = None,
                         entity_col: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = self.df.copy()
        if time_col is not None and entity_col is not None:
            for col in [outcome_col, treat_col] + feature_cols:
                df[f"{col}_demeaned"] = df.groupby(entity_col)[col].transform(lambda x: x - x.mean())
            Y = df[f"{outcome_col}_demeaned"].values
            T = df[f"{treat_col}_demeaned"].values
            X = df[[f"{c}_demeaned" for c in feature_cols]].values
        else:
            Y = df[outcome_col].values
            T = df[treat_col].values
            X = df[feature_cols].values
        mask = ~(np.isnan(Y) | np.isnan(T) | np.isnan(X).any(axis=1))
        Y, T, X = Y[mask], T[mask], X[mask]
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
        logger.info(f"DML数据准备完成：{len(Y)}个样本，{X.shape[1]}个特征")
        return Y, T, X


# ========== 传统计量分析模块 ==========
class CausalInferenceAnalyzer:
    @staticmethod
    def psm_matching(df: pd.DataFrame, treat_col: str, match_vars: List[str], caliper=0.05) -> pd.DataFrame:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = df[match_vars].fillna(df[match_vars].median())
        y = df[treat_col]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logit = LogisticRegression(max_iter=1000)
        logit.fit(X_scaled, y)
        df = df.copy()
        df["pscore"] = logit.predict_proba(X_scaled)[:, 1]
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

    @staticmethod
    def did_regression(df: pd.DataFrame, outcome: str, did_col="did", treat_col="treat",
                       post_col="post", controls: List[str] = None,
                       entity_col="企业代码", time_col="年份") -> Dict:
        try:
            from linearmodels.panel import PanelOLS
            df_panel = df.set_index([entity_col, time_col])
            ctrl_str = " + ".join(controls) if controls else ""
            formula = f"{outcome} ~ {did_col} + {ctrl_str} + EntityEffects + TimeEffects"
            model = PanelOLS.from_formula(formula, data=df_panel, drop_absorbed=True)
            res = model.fit(cov_type="clustered", cluster_entity=True)
        except Exception:
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

    @staticmethod
    def rdd_regression(df: pd.DataFrame, outcome: str, running="running_centered",
                       treated="treated", poly_order=1) -> Dict:
        df = df.copy()
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

    @staticmethod
    def parallel_trend_test(df: pd.DataFrame, outcome: str, treat_col: str,
                            base_year: int, time_col="年份") -> Dict:
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

    @staticmethod
    def plot_parallel_trend(parallel_trend_result: Dict, policy_year: int, save_path="parallel_trend.png"):
        years = parallel_trend_result["years"]
        coefs = parallel_trend_result["coefficients"]
        ci_low = parallel_trend_result["ci_low"]
        ci_high = parallel_trend_result["ci_high"]
        plt.figure(figsize=(10, 6))
        plt.errorbar(years, coefs, yerr=[(coefs[i]-ci_low[i]) for i in range(len(coefs))],
                     fmt='o', capsize=5, color='steelblue', ecolor='gray', elinewidth=2)
        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='零线')
        plt.axvline(x=str(policy_year), color='gray', linestyle='--', linewidth=1.5, label=f'政策实施年 ({policy_year})')
        plt.xlabel("年份", fontsize=12)
        plt.ylabel("处理组与对照组差异（系数）", fontsize=12)
        plt.title("平行趋势检验", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"平行趋势图已保存至 {save_path}")


# ========== 双重机器学习模块（稳定版） ==========
class DoubleMachineLearningAnalyzer:
    def __init__(self, model_y=None, model_t=None, n_folds=5):
        self.model_y = model_y
        self.model_t = model_t
        self.n_folds = n_folds
        self._fitted = False
        self._ate = None
        self._ate_std = None
        self._ate_pvalue = None
        self._ate_ci = None

    def _get_default_models(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import RidgeCV
            model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42)
            model_t = RidgeCV(alphas=[0.1, 1.0, 10.0])
            return model_y, model_t
        except ImportError:
            from sklearn.linear_model import LinearRegression, RidgeCV
            return LinearRegression(), RidgeCV(alphas=[0.1, 1.0, 10.0])

    def _t_stat_pvalue(self, t_value, df=100):
        from scipy import stats
        return 2 * (1 - stats.t.cdf(abs(t_value), df=df))

    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray):
        from sklearn.model_selection import KFold
        if self.model_y is None or self.model_t is None:
            self.model_y, self.model_t = self._get_default_models()

        n = len(Y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        Y_residuals = np.zeros(n)
        T_residuals = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            T_train, T_val = T[train_idx], T[val_idx]
            self.model_y.fit(X_train, Y_train)
            self.model_t.fit(X_train, T_train)
            Y_pred = self.model_y.predict(X_val)
            T_pred = self.model_t.predict(X_val)
            Y_residuals[val_idx] = Y_val - Y_pred
            T_residuals[val_idx] = T_val - T_pred

        denominator = np.sum(T_residuals ** 2)
        if denominator < 1e-10:
            logger.warning("T残差平方和接近于0，添加微小噪声继续估计")
            T_residuals += np.random.normal(0, 1e-8, size=len(T_residuals))
            denominator = np.sum(T_residuals ** 2)

        theta = np.sum(Y_residuals * T_residuals) / denominator
        residual = Y_residuals - theta * T_residuals
        sigma2_hat = np.mean(residual ** 2)
        var_theta = sigma2_hat / denominator
        std_theta = np.sqrt(var_theta)

        self._ate = theta
        self._ate_std = std_theta
        df_approx = max(10, n - X.shape[1])
        self._ate_pvalue = self._t_stat_pvalue(theta / std_theta, df=df_approx)
        self._ate_ci = (theta - 1.96 * std_theta, theta + 1.96 * std_theta)
        self._fitted = True
        logger.info(f"DML估计完成：ATE={theta:.4f} (se={std_theta:.4f}), p={self._ate_pvalue:.4f}")
        return self

    def get_ate_results(self) -> Dict:
        if not self._fitted:
            raise ValueError("模型尚未拟合")
        return {
            "ate_coefficient": round(float(self._ate), 4) if not np.isnan(self._ate) else np.nan,
            "std_error": round(float(self._ate_std), 4) if not np.isnan(self._ate_std) else np.nan,
            "p_value": round(float(self._ate_pvalue), 4) if not np.isnan(self._ate_pvalue) else np.nan,
            "significant": self._ate_pvalue < 0.05 if not np.isnan(self._ate_pvalue) else False,
            "ci_lower": round(float(self._ate_ci[0]), 4) if not np.isnan(self._ate_ci[0]) else np.nan,
            "ci_upper": round(float(self._ate_ci[1]), 4) if not np.isnan(self._ate_ci[1]) else np.nan,
            "method": "Double Machine Learning (DML)",
        }


# ========== LLM解读模块（使用通义千问引擎） ==========
class PolicyEvalInterpreter:
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def interpret_did(self, did_result: Dict, policy_name: str, outcome_name: str,
                      parallel_trend_result: Dict = None, policy_year: int = None) -> str:
        # 准备平行趋势描述
        pt_description = ""
        if parallel_trend_result and "years" in parallel_trend_result:
            years = parallel_trend_result["years"]
            ci_low = parallel_trend_result["ci_low"]
            ci_high = parallel_trend_result["ci_high"]
            if policy_year is not None:
                pre_indices = [i for i, y in enumerate(years) if int(y) < policy_year]
            else:
                pre_indices = list(range(len(years)))
            any_significant = False
            for i in pre_indices:
                if ci_low[i] > 0 or ci_high[i] < 0:
                    any_significant = True
                    break
            if any_significant:
                pt_description = "⚠️ 平行趋势假设可能不成立：政策前某些年份处理组与对照组差异显著，DID估计可能存在偏误。"
            else:
                pt_description = "✅ 平行趋势假设成立：政策前各期处理组与对照组差异不显著，DID估计可信。"
        else:
            pt_description = "未进行平行趋势检验或数据不足。"

        prompt = f"""
你是专业的政策评估研究员。请基于以下双重差分（DID）分析结果和平行趋势检验，撰写一份政府研究报告风格的政策解读，要求：
1. **政策效果总结**：说明政策对{outcome_name}的影响方向、大小（DID系数）以及统计显著性。如果显著，量化效应量；如果不显著，分析可能原因。
2. **平行趋势评估**：根据平行趋势检验结果，判断DID方法的前提是否满足，并说明对结论可信度的影响。
3. **政策建议**：结合政策效果和平行趋势结论，提出3条具体、可操作的政策优化建议或后续研究方向。

政策名称：{policy_name}
结果变量：{outcome_name}
DID系数：{did_result['did_coefficient']}
P值：{did_result['p_value']}（{'显著' if did_result['significant'] else '不显著'}）
样本量：{did_result['n_obs']}
R²：{did_result['r_squared']}
平行趋势检验结果：{pt_description}

请输出300-500字，语言正式、简洁，分三段（政策效果、平行趋势评估、政策建议）。
"""
        return self.llm.generate(
            "你是专业政策评估研究员，擅长将计量结果转化为政策语言。",
            prompt, max_tokens=800, temperature=0.4
        )

    def interpret_dml(self, dml_result: Dict, policy_name: str, outcome_name: str) -> str:
        prompt = f"""
你是专业的政策评估研究员。请基于以下双重机器学习（DML）估计结果，撰写政策解读，要求：
1. 解释平均处理效应（ATE）的大小、方向和统计显著性。
2. 说明DML方法相比传统DID的优势（如控制高维协变量、捕捉非线性关系）。
3. 提出3条针对该政策的优化建议或未来评估方向。

政策名称：{policy_name}
结果变量：{outcome_name}
DML-ATE系数：{dml_result.get('ate_coefficient', 'N/A')}
标准误：{dml_result.get('std_error', 'N/A')}
P值：{dml_result.get('p_value', 'N/A')}（{'显著' if dml_result.get('significant', False) else '不显著'}）
95%置信区间：[{dml_result.get('ci_lower', 'N/A')}, {dml_result.get('ci_upper', 'N/A')}]
估计方法：{dml_result.get('method', 'DML')}

请输出300-500字。
"""
        return self.llm.generate(
            "你是专业政策评估研究员，擅长将机器学习因果推断结果转化为政策语言。",
            prompt, max_tokens=800, temperature=0.4
        )


# ========== 全流程整合 ==========
class PolicyEvaluationPipeline:
    def __init__(self, data_path: str, llm_engine=None):
        self.preparer = PolicyDataPreparer(data_path)
        self.analyzer = CausalInferenceAnalyzer()
        self.dml_analyzer = DoubleMachineLearningAnalyzer()
        self.interpreter = PolicyEvalInterpreter(llm_engine) if llm_engine else None

    def run_did_evaluation(self, policy_name: str, policy_year: int,
                           treat_col: str, outcome_col: str,
                           control_cols: List[str], match_vars: List[str]) -> Dict:
        logger.info(f"开始PSM-DID评估政策：{policy_name}")
        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)
        df_matched = self.analyzer.psm_matching(df, treat_col, match_vars)

        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df_matched.columns else outcome_col
        did_res = self.analyzer.did_regression(
            df_matched, outcome_ln,
            controls=[f"ln_{c}" if f"ln_{c}" in df_matched.columns else c for c in control_cols]
        )
        did_res["policy_year"] = policy_year

        pt_res = self.analyzer.parallel_trend_test(df_matched, outcome_ln, treat_col, policy_year - 1)
        self.analyzer.plot_parallel_trend(pt_res, policy_year, save_path=f"parallel_trend_{policy_name[:10]}.png")

        interpretation = ""
        if self.interpreter:
            interpretation = self.interpreter.interpret_did(
                did_res, policy_name, outcome_col,
                parallel_trend_result=pt_res,
                policy_year=policy_year
            )

        result = {
            "policy_name": policy_name,
            "method": "PSM-DID",
            "did_results": did_res,
            "parallel_trend": pt_res,
            "llm_interpretation": interpretation,
        }
        self._save_results(result, f"eval_{policy_name[:10]}_did")
        return result

    def run_dml_evaluation(self, policy_name: str, outcome_col: str, treat_col: str,
                           feature_cols: List[str], time_col: str = None, entity_col: str = None) -> Dict:
        logger.info(f"开始DML评估政策：{policy_name}")
        Y, T, X = self.preparer.prepare_dml_data(outcome_col, treat_col, feature_cols, time_col, entity_col)
        self.dml_analyzer.fit(Y, T, X)
        dml_res = self.dml_analyzer.get_ate_results()
        interpretation = ""
        if self.interpreter:
            interpretation = self.interpreter.interpret_dml(dml_res, policy_name, outcome_col)
        result = {
            "policy_name": policy_name,
            "method": "DML",
            "dml_results": dml_res,
            "sample_size": len(Y),
            "n_features": X.shape[1],
            "llm_interpretation": interpretation,
        }
        self._save_results(result, f"eval_{policy_name[:10]}_dml")
        logger.info(f"DML评估完成：ATE={dml_res['ate_coefficient']:.4f}, p={dml_res['p_value']:.4f}")
        return result

    def run_comparison_evaluation(self, policy_name: str, policy_year: int,
                                   treat_col: str, outcome_col: str,
                                   control_cols: List[str], match_vars: List[str],
                                   feature_cols: List[str],
                                   time_col: str = "年份", entity_col: str = "企业代码") -> Dict:
        logger.info(f"开始对比评估政策：{policy_name}")
        did_result = self.run_did_evaluation(policy_name, policy_year, treat_col, outcome_col,
                                              control_cols, match_vars)
        dml_result = self.run_dml_evaluation(policy_name, outcome_col, treat_col, feature_cols,
                                             time_col, entity_col)
        comparison = {
            "policy_name": policy_name,
            "did_ate": did_result["did_results"]["did_coefficient"],
            "did_pvalue": did_result["did_results"]["p_value"],
            "did_significant": did_result["did_results"]["significant"],
            "dml_ate": dml_result["dml_results"]["ate_coefficient"],
            "dml_pvalue": dml_result["dml_results"]["p_value"],
            "dml_significant": dml_result["dml_results"]["significant"],
            "dml_ci": [dml_result["dml_results"]["ci_lower"], dml_result["dml_results"]["ci_upper"]],
            "conclusion": self._generate_comparison_conclusion(did_result, dml_result),
        }
        self._save_results(comparison, f"eval_{policy_name[:10]}_comparison")
        return {"did": did_result, "dml": dml_result, "comparison": comparison}

    def _generate_comparison_conclusion(self, did_result: Dict, dml_result: Dict) -> str:
        did_ate = did_result["did_results"]["did_coefficient"]
        dml_ate = dml_result["dml_results"]["ate_coefficient"]
        did_sig = did_result["did_results"]["significant"]
        dml_sig = dml_result["dml_results"]["significant"]
        if np.isnan(dml_ate):
            return "DML估计失败（可能由于处理变量与协变量完全线性相关），建议检查数据或使用其他方法。"
        if did_sig and dml_sig:
            if abs(did_ate - dml_ate) / (abs(dml_ate) + 1e-6) < 0.2:
                return f"两种方法均显著，效应量一致（DID={did_ate:.3f}, DML={dml_ate:.3f}），政策效果稳健。"
            else:
                return f"两种方法均显著但效应量存在差异，DML控制了更多协变量，建议以DML结果为准。"
        elif did_sig and not dml_sig:
            return f"DID显著但DML不显著，可能存在遗漏变量偏误，DML结果更可信。"
        elif not did_sig and dml_sig:
            return f"DML显著而DID不显著，表明DML在高维非线性关系下优势明显。"
        else:
            return "两种方法均不显著，未能检测到稳健的政策效应。"

    def _save_results(self, result: Dict, filename: str):
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return make_json_serializable(obj.tolist())
            elif isinstance(obj, pd.Series):
                return make_json_serializable(obj.to_dict())
            elif isinstance(obj, pd.DataFrame):
                return make_json_serializable(obj.to_dict(orient='records'))
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj
        serializable_result = make_json_serializable(result)
        with open(f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)


# ========== 模拟数据与运行示例 ==========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    np.random.seed(42)

    # 生成模拟数据（与之前相同）
    n = 500
    size = np.random.lognormal(11, 1.2, n)
    revenue = size * np.random.uniform(0.5, 2, n)
    employees = size * np.random.uniform(0.001, 0.01, n)
    roa = np.random.normal(0.05, 0.03, n)
    leverage = np.random.uniform(0.3, 0.8, n)
    cashflow = revenue * np.random.uniform(0.05, 0.2, n)

    propensity = 0.5 + 0.1 * (size - size.mean()) / size.std() + 0.1 * (revenue - revenue.mean()) / revenue.std()
    propensity = np.clip(propensity, 0.1, 0.9)
    treat = np.random.binomial(1, propensity, n)

    year = np.tile([2018, 2019, 2020, 2021, 2022], 100)
    base_rd = np.exp(10 + 0.5 * np.log(size) + 0.2 * np.log(revenue) + np.random.normal(0, 0.3, n))
    policy_effect = 0.15 * treat * (year >= 2020)
    rd = base_rd * (1 + policy_effect)

    mock_data = pd.DataFrame({
        "企业代码": np.repeat(range(100), 5),
        "年份": year,
        "treat": treat,
        "研发投入": rd,
        "营业收入": revenue,
        "资产总额": size,
        "员工人数": employees,
        "ROA": roa,
        "资产负债率": leverage,
        "现金流": cashflow,
    })
    mock_data.to_csv("mock_panel_data.csv", index=False, encoding="utf-8-sig")

    # 初始化通义千问引擎（请替换为您的 API Key）
    # 方式一：直接传入 API Key（不推荐硬编码，这里仅为示例）
    # llm_engine = QwenEngine(api_key="sk-2f95ab436b644f11849c067a74744c7a", model="qwen-turbo")
    # 方式二：从环境变量读取（推荐）
    import os
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-2f95ab436b644f11849c067a74744c7a")  # 默认使用您提供的 key
    llm_engine = QwenEngine(api_key=api_key, model="qwen-turbo")

    pipeline = PolicyEvaluationPipeline("mock_panel_data.csv", llm_engine=llm_engine)

    print("\n" + "=" * 60)
    print("【方法一】传统PSM-DID评估")
    print("=" * 60)
    did_result = pipeline.run_did_evaluation(
        policy_name="专精特新企业认定政策",
        policy_year=2020,
        treat_col="treat",
        outcome_col="研发投入",
        control_cols=["营业收入", "资产总额", "员工人数"],
        match_vars=["营业收入", "资产总额", "员工人数"],
    )
    print(f"DID系数：{did_result['did_results']['did_coefficient']}")
    print(f"P值：{did_result['did_results']['p_value']}")
    print(f"是否显著：{did_result['did_results']['significant']}")
    print(f"\nLLM解读：\n{did_result['llm_interpretation']}")

    print("\n" + "=" * 60)
    print("【方法二】双重机器学习（DML）评估")
    print("=" * 60)
    dml_result = pipeline.run_dml_evaluation(
        policy_name="专精特新企业认定政策",
        outcome_col="研发投入",
        treat_col="treat",
        feature_cols=["营业收入", "资产总额", "员工人数", "ROA", "资产负债率", "现金流"],
        time_col="年份",
        entity_col="企业代码",
    )
    print(f"DML-ATE系数：{dml_result['dml_results']['ate_coefficient']}")
    print(f"标准误：{dml_result['dml_results']['std_error']}")
    print(f"95%置信区间：[{dml_result['dml_results']['ci_lower']}, {dml_result['dml_results']['ci_upper']}]")
    print(f"是否显著：{dml_result['dml_results']['significant']}")
    print(f"\nLLM解读：\n{dml_result['llm_interpretation']}")

    print("\n" + "=" * 60)
    print("【方法三】方法对比评估（DID vs DML）")
    print("=" * 60)
    comparison = pipeline.run_comparison_evaluation(
        policy_name="专精特新企业认定政策",
        policy_year=2020,
        treat_col="treat",
        outcome_col="研发投入",
        control_cols=["营业收入", "资产总额", "员工人数"],
        match_vars=["营业收入", "资产总额", "员工人数"],
        feature_cols=["营业收入", "资产总额", "员工人数", "ROA", "资产负债率", "现金流"],
        time_col="年份",
        entity_col="企业代码",
    )
    print(f"对比结论：{comparison['comparison']['conclusion']}")