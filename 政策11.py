"""
政策效果自动评估流水线（三种方法对比：普通DID、PSM-DID、DML）
支持自动列名适配 + 平行趋势检验 + 通义千问 LLM 解读
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


# ========== 通义千问 LLM 引擎 ==========
class QwenEngine:
    def __init__(self, api_key: str = None, model: str = "qwen-turbo"):
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
        full_prompt = f"系统指令：{system_prompt}\n\n用户问题：{user_prompt}"
        return self._safe_api_call(full_prompt, max_tokens, temperature)

    def _safe_api_call(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            print(f"调用通义千问 API，提示长度: {len(prompt)}", file=sys.stderr)
            messages = [
                {"role": "system",
                 "content": "你是一个专业的政策评估研究员，擅长将计量经济学结果转化为政府研究报告风格的语言。请用中文回答。"},
                {"role": "user", "content": prompt}
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
                return f"API 调用失败: [{response.code}] {response.message}"
            if hasattr(response, 'output') and response.output:
                if hasattr(response.output, 'choices') and response.output.choices:
                    if len(response.output.choices) > 0:
                        choice = response.output.choices[0]
                        if hasattr(choice, 'message') and choice.message:
                            if hasattr(choice.message, 'content'):
                                content = choice.message.content
                                print(f"成功获取响应，长度: {len(content)}", file=sys.stderr)
                                return content
            if hasattr(response, 'output') and hasattr(response.output, 'text'):
                return response.output.text
            return "根据计量分析结果，未能生成有效的政策解读。"
        except Exception as e:
            print(f"API 调用异常: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return f"系统错误: {str(e)}"


# ========== 数据准备模块（自动适配列名） ==========
class PolicyDataPreparer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path, encoding="utf-8-sig")
        self.df.columns = [col.strip() for col in self.df.columns]
        logger.info(f"数据加载成功：{self.df.shape[0]}行 × {self.df.shape[1]}列")

        # 自动识别列名映射
        self.col_map = {}
        # 时间列
        if "year" in self.df.columns:
            self.col_map["time"] = "year"
        elif "年份" in self.df.columns:
            self.col_map["time"] = "年份"
        else:
            raise ValueError("数据中缺少时间列（需要 'year' 或 '年份'）")

        # 实体列
        if "id" in self.df.columns:
            self.col_map["entity"] = "id"
        elif "企业代码" in self.df.columns:
            self.col_map["entity"] = "企业代码"
        else:
            raise ValueError("数据中缺少实体列（需要 'id' 或 '企业代码'）")

        # 处理组列（真实数据用 did，模拟数据用 treat）
        if "did" in self.df.columns:
            self.col_map["treat"] = "did"
        elif "treat" in self.df.columns:
            self.col_map["treat"] = "treat"
        else:
            raise ValueError("数据中缺少处理组列（需要 'did' 或 'treat'）")

        logger.info(
            f"列名映射: time={self.col_map['time']}, entity={self.col_map['entity']}, treat={self.col_map['treat']}")

    def prepare_did_data(self, policy_year: int, treat_col: str,
                         outcome_col: str, control_cols: List[str]) -> pd.DataFrame:
        """
        准备 DID 数据，自动使用内部映射的列名
        """
        df = self.df.copy()
        actual_time_col = self.col_map["time"]
        actual_treat_col = treat_col

        df["post"] = (df[actual_time_col] >= policy_year).astype(int)
        df["did_interaction"] = df[actual_treat_col] * df["post"]

        # 对连续变量取对数
        for col in [outcome_col] + control_cols:
            if col in df.columns and df[col].min() > 0:
                df[f"ln_{col}"] = np.log(df[col])
        return df

    def prepare_dml_data(self, outcome_col: str, treat_col: str,
                         feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备 DML 数据，支持面板去均值
        """
        df = self.df.copy()
        actual_entity_col = self.col_map["entity"]
        actual_treat_col = treat_col

        # 面板去均值（固定效应）
        for col in [outcome_col, actual_treat_col] + feature_cols:
            if col in df.columns:
                df[f"{col}_demeaned"] = df.groupby(actual_entity_col)[col].transform(lambda x: x - x.mean())

        Y = df[f"{outcome_col}_demeaned"].values
        T = df[f"{actual_treat_col}_demeaned"].values
        X = df[[f"{c}_demeaned" for c in feature_cols if c in df.columns]].values

        mask = ~(np.isnan(Y) | np.isnan(T) | np.isnan(X).any(axis=1))
        Y, T, X = Y[mask], T[mask], X[mask]
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
        logger.info(f"DML数据准备完成：{len(Y)}个样本，{X.shape[1]}个特征")
        return Y, T, X


# ========== 计量分析模块 ==========
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
    def did_regression(df: pd.DataFrame, outcome: str,
                       did_col="did_interaction", treat_col="treat",
                       post_col="post", controls: List[str] = None,
                       entity_col="id", time_col="year") -> Dict:
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
        did_coef = res.params.get(did_col, np.nan)
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
    def parallel_trend_test(df: pd.DataFrame, outcome: str, treat_col: str,
                            base_year: int, time_col="year") -> Dict:
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
        plt.errorbar(years, coefs, yerr=[(coefs[i] - ci_low[i]) for i in range(len(coefs))],
                     fmt='o', capsize=5, color='steelblue', ecolor='gray', elinewidth=2)
        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='零线')
        plt.axvline(x=str(policy_year), color='gray', linestyle='--', linewidth=1.5,
                    label=f'政策实施年 ({policy_year})')
        plt.xlabel("年份", fontsize=12)
        plt.ylabel("处理组与对照组差异（系数）", fontsize=12)
        plt.title("平行趋势检验", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"平行趋势图已保存至 {save_path}")


# ========== 双重机器学习模块 ==========
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
            return RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42), \
                RidgeCV(alphas=[0.1, 1.0, 10.0])
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
        Y_res = np.zeros(n)
        T_res = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            T_train, T_val = T[train_idx], T[val_idx]
            self.model_y.fit(X_train, Y_train)
            self.model_t.fit(X_train, T_train)
            Y_pred = self.model_y.predict(X_val)
            T_pred = self.model_t.predict(X_val)
            Y_res[val_idx] = Y_val - Y_pred
            T_res[val_idx] = T_val - T_pred

        denom = np.sum(T_res ** 2)
        if denom < 1e-10:
            logger.warning("T残差平方和过小，添加噪声")
            T_res += np.random.normal(0, 1e-8, size=n)
            denom = np.sum(T_res ** 2)

        theta = np.sum(Y_res * T_res) / denom
        resid = Y_res - theta * T_res
        sigma2 = np.mean(resid ** 2)
        var_theta = sigma2 / denom
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
            raise ValueError("模型未拟合")
        return {
            "ate_coefficient": round(float(self._ate), 4) if not np.isnan(self._ate) else np.nan,
            "std_error": round(float(self._ate_std), 4) if not np.isnan(self._ate_std) else np.nan,
            "p_value": round(float(self._ate_pvalue), 4) if not np.isnan(self._ate_pvalue) else np.nan,
            "significant": self._ate_pvalue < 0.05 if not np.isnan(self._ate_pvalue) else False,
            "ci_lower": round(float(self._ate_ci[0]), 4) if not np.isnan(self._ate_ci[0]) else np.nan,
            "ci_upper": round(float(self._ate_ci[1]), 4) if not np.isnan(self._ate_ci[1]) else np.nan,
            "method": "Double Machine Learning (DML)",
        }


# ========== LLM 解读模块 ==========
class PolicyEvalInterpreter:
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def interpret_did(self, method_name: str, did_result: Dict, policy_name: str, outcome_name: str,
                      parallel_trend_result: Dict = None, policy_year: int = None) -> str:
        pt_desc = "未进行平行趋势检验"
        if parallel_trend_result and "years" in parallel_trend_result:
            years = parallel_trend_result["years"]
            ci_low = parallel_trend_result["ci_low"]
            ci_high = parallel_trend_result["ci_high"]
            if policy_year:
                pre_idx = [i for i, y in enumerate(years) if int(y) < policy_year]
                any_sig = any(ci_low[i] > 0 or ci_high[i] < 0 for i in pre_idx)
                pt_desc = "⚠️ 平行趋势假设可能不成立" if any_sig else "✅ 平行趋势假设成立"
        prompt = f"""
你是专业的政策评估研究员。请基于{method_name}方法的DID分析结果和平行趋势检验，撰写政府研究报告风格的政策解读，分三段：
1. 政策效果总结（方向、大小、显著性）
2. 平行趋势评估
3. 三条具体政策建议

方法名称：{method_name}
政策名称：{policy_name}
结果变量：{outcome_name}
DID系数：{did_result['did_coefficient']}
P值：{did_result['p_value']}（{'显著' if did_result['significant'] else '不显著'}）
样本量：{did_result['n_obs']}
R²：{did_result['r_squared']}
平行趋势结论：{pt_desc}

输出300-500字。
"""
        return self.llm.generate("政策评估专家", prompt)

    def interpret_dml(self, dml_result: Dict, policy_name: str, outcome_name: str) -> str:
        prompt = f"""
请基于DML估计结果撰写政策解读：
- ATE系数：{dml_result.get('ate_coefficient', 'N/A')}
- P值：{dml_result.get('p_value', 'N/A')}（{'显著' if dml_result.get('significant', False) else '不显著'}）
- 95%置信区间：[{dml_result.get('ci_lower', 'N/A')}, {dml_result.get('ci_upper', 'N/A')}]
要求：解释ATE含义、DML优势、三条优化建议。300-500字。
"""
        return self.llm.generate("政策评估专家", prompt)


# ========== 流水线整合（三种方法） ==========
class PolicyEvaluationPipeline:
    def __init__(self, data_path: str, llm_engine=None):
        self.preparer = PolicyDataPreparer(data_path)
        self.analyzer = CausalInferenceAnalyzer()
        self.dml_analyzer = DoubleMachineLearningAnalyzer()
        self.interpreter = PolicyEvalInterpreter(llm_engine) if llm_engine else None

    def run_naive_did(self, policy_name: str, policy_year: int,
                      treat_col: str, outcome_col: str,
                      control_cols: List[str]) -> Dict:
        """
        普通 DID（不进行 PSM 匹配）
        """
        logger.info(f"开始普通DID评估：{policy_name}")
        actual_time_col = self.preparer.col_map["time"]
        actual_entity_col = self.preparer.col_map["entity"]

        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)

        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df.columns else outcome_col
        ctrl_ln = [f"ln_{c}" if f"ln_{c}" in df.columns else c for c in control_cols]

        did_res = self.analyzer.did_regression(
            df, outcome_ln,
            did_col="did_interaction",
            treat_col=treat_col,
            post_col="post",
            controls=ctrl_ln,
            entity_col=actual_entity_col,
            time_col=actual_time_col
        )
        did_res["policy_year"] = policy_year

        # 平行趋势检验
        pt_res = self.analyzer.parallel_trend_test(df, outcome_ln, treat_col, policy_year - 1, actual_time_col)
        self.analyzer.plot_parallel_trend(pt_res, policy_year, f"parallel_trend_naive_{policy_name}.png")

        interpretation = ""
        if self.interpreter:
            interpretation = self.interpreter.interpret_did(
                "普通DID", did_res, policy_name, outcome_col, pt_res, policy_year
            )

        result = {
            "method": "普通DID",
            "did_results": did_res,
            "parallel_trend": pt_res,
            "llm_interpretation": interpretation,
        }
        self._save_results(result, f"eval_naive_{policy_name}")
        return result

    def run_psm_did(self, policy_name: str, policy_year: int,
                    treat_col: str, outcome_col: str,
                    control_cols: List[str], match_vars: List[str]) -> Dict:
        """
        PSM-DID（倾向得分匹配 + DID）
        """
        logger.info(f"开始PSM-DID评估：{policy_name}")
        actual_time_col = self.preparer.col_map["time"]
        actual_entity_col = self.preparer.col_map["entity"]

        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)
        df_matched = self.analyzer.psm_matching(df, treat_col, match_vars)

        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df_matched.columns else outcome_col
        ctrl_ln = [f"ln_{c}" if f"ln_{c}" in df_matched.columns else c for c in control_cols]

        did_res = self.analyzer.did_regression(
            df_matched, outcome_ln,
            did_col="did_interaction",
            treat_col=treat_col,
            post_col="post",
            controls=ctrl_ln,
            entity_col=actual_entity_col,
            time_col=actual_time_col
        )
        did_res["policy_year"] = policy_year

        pt_res = self.analyzer.parallel_trend_test(df_matched, outcome_ln, treat_col, policy_year - 1, actual_time_col)
        self.analyzer.plot_parallel_trend(pt_res, policy_year, f"parallel_trend_psm_{policy_name}.png")

        interpretation = ""
        if self.interpreter:
            interpretation = self.interpreter.interpret_did(
                "PSM-DID", did_res, policy_name, outcome_col, pt_res, policy_year
            )

        result = {
            "method": "PSM-DID",
            "did_results": did_res,
            "parallel_trend": pt_res,
            "llm_interpretation": interpretation,
        }
        self._save_results(result, f"eval_psm_{policy_name}")
        return result

    def run_dml(self, policy_name: str, outcome_col: str, treat_col: str,
                feature_cols: List[str]) -> Dict:
        """
        双重机器学习（DML）
        """
        logger.info(f"开始DML评估：{policy_name}")
        Y, T, X = self.preparer.prepare_dml_data(outcome_col, treat_col, feature_cols)
        self.dml_analyzer.fit(Y, T, X)
        dml_res = self.dml_analyzer.get_ate_results()
        interpretation = ""
        if self.interpreter:
            interpretation = self.interpreter.interpret_dml(dml_res, policy_name, outcome_col)
        result = {
            "method": "DML",
            "dml_results": dml_res,
            "sample_size": len(Y),
            "n_features": X.shape[1],
            "llm_interpretation": interpretation,
        }
        self._save_results(result, f"eval_dml_{policy_name}")
        logger.info(f"DML评估完成：ATE={dml_res['ate_coefficient']:.4f}, p={dml_res['p_value']:.4f}")
        return result

    def run_full_comparison(self, policy_name: str, policy_year: int,
                            treat_col: str, outcome_col: str,
                            control_cols: List[str], match_vars: List[str],
                            dml_features: List[str]) -> Dict:
        """
        完整对比：普通DID、PSM-DID、DML
        """
        logger.info(f"开始三种方法对比评估：{policy_name}")

        # 1. 普通DID
        naive_result = self.run_naive_did(policy_name, policy_year, treat_col, outcome_col, control_cols)

        # 2. PSM-DID
        psm_result = self.run_psm_did(policy_name, policy_year, treat_col, outcome_col, control_cols, match_vars)

        # 3. DML
        dml_result = self.run_dml(policy_name, outcome_col, treat_col, dml_features)

        # 构建对比表格
        comparison = {
            "policy_name": policy_name,
            "policy_year": policy_year,
            "naive_did": {
                "coefficient": naive_result["did_results"]["did_coefficient"],
                "p_value": naive_result["did_results"]["p_value"],
                "significant": naive_result["did_results"]["significant"],
                "n_obs": naive_result["did_results"]["n_obs"],
            },
            "psm_did": {
                "coefficient": psm_result["did_results"]["did_coefficient"],
                "p_value": psm_result["did_results"]["p_value"],
                "significant": psm_result["did_results"]["significant"],
                "n_obs": psm_result["did_results"]["n_obs"],
            },
            "dml": {
                "coefficient": dml_result["dml_results"]["ate_coefficient"],
                "std_error": dml_result["dml_results"]["std_error"],
                "p_value": dml_result["dml_results"]["p_value"],
                "significant": dml_result["dml_results"]["significant"],
                "ci_lower": dml_result["dml_results"]["ci_lower"],
                "ci_upper": dml_result["dml_results"]["ci_upper"],
            },
            "conclusion": self._generate_comparison_conclusion(naive_result, psm_result, dml_result),
        }

        self._save_results(comparison, f"comparison_{policy_name}")
        return {
            "naive_did": naive_result,
            "psm_did": psm_result,
            "dml": dml_result,
            "comparison": comparison,
        }

    def _generate_comparison_conclusion(self, naive: Dict, psm: Dict, dml: Dict) -> str:
        naive_coef = naive["did_results"]["did_coefficient"]
        psm_coef = psm["did_results"]["did_coefficient"]
        dml_coef = dml["dml_results"]["ate_coefficient"]

        naive_sig = naive["did_results"]["significant"]
        psm_sig = psm["did_results"]["significant"]
        dml_sig = dml["dml_results"]["significant"]

        conclusion = f"普通DID估计值为{naive_coef:.3f}（{'显著' if naive_sig else '不显著'}），"
        conclusion += f"PSM-DID估计值为{psm_coef:.3f}（{'显著' if psm_sig else '不显著'}），"
        conclusion += f"DML估计值为{dml_coef:.3f}（{'显著' if dml_sig else '不显著'}）。"

        if dml_sig and (naive_sig or psm_sig):
            conclusion += " DML结果显著且控制了高维协变量，结论更可信。"
        elif (naive_sig or psm_sig) and not dml_sig:
            conclusion += " 传统DID显著但DML不显著，可能存在遗漏变量或非线性关系，建议以DML为准。"
        elif dml_sig and not (naive_sig or psm_sig):
            conclusion += " DML捕捉到了传统方法未能发现的因果效应，体现了机器学习优势。"
        else:
            conclusion += " 三种方法均不显著，政策效果可能有限或数据不足。"
        return conclusion

    def _save_results(self, result: Dict, filename: str):
        def convert(obj):
            if isinstance(obj, dict):
                return {convert(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(i) for i in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return convert(obj.tolist())
            elif isinstance(obj, pd.Series):
                return convert(obj.to_dict())
            elif isinstance(obj, pd.DataFrame):
                return convert(obj.to_dict(orient='records'))
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj

        with open(f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(convert(result), f, ensure_ascii=False, indent=2)


# ========== 主程序 ==========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 真实数据路径
    DATA_PATH = "data.csv"  # 您的真实数据文件

    # 初始化通义千问（使用环境变量或直接填写 API Key）
    import os

    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-2f95ab436b644f11849c067a74744c7a")
    llm_engine = QwenEngine(api_key=api_key, model="qwen-turbo")

    pipeline = PolicyEvaluationPipeline(DATA_PATH, llm_engine=llm_engine)

    # ========== 真实数据参数设置（请根据您的数据修改） ==========
    POLICY_NAME = "低碳试点政策"
    POLICY_YEAR = 2010          # 请根据实际政策实施年份修改
    TREAT_COL = "did"           # 处理组标识列
    OUTCOME_COL = "hfd5"        # 结果变量（绿色创新指标）
    CONTROL_COLS = ["Size", "Lev", "ROA", "Emply", "Age"]
    MATCH_VARS = ["Size", "Lev", "ROA", "Emply", "Age"]
    DML_FEATURES = ["Size", "Lev", "ROA", "Emply", "Age"]
    # ===========================================================

    # 运行三种方法完整对比
    comparison = pipeline.run_full_comparison(
        policy_name=POLICY_NAME,
        policy_year=POLICY_YEAR,
        treat_col=TREAT_COL,
        outcome_col=OUTCOME_COL,
        control_cols=CONTROL_COLS,
        match_vars=MATCH_VARS,
        dml_features=DML_FEATURES,
    )

    # 打印对比结果
    print("\n" + "=" * 60)
    print("【三种政策评估方法对比】")
    print("=" * 60)
    print(f"普通DID系数: {comparison['naive_did']['did_results']['did_coefficient']} "
          f"(p={comparison['naive_did']['did_results']['p_value']}, "
          f"{'显著' if comparison['naive_did']['did_results']['significant'] else '不显著'})")
    print(f"PSM-DID系数: {comparison['psm_did']['did_results']['did_coefficient']} "
          f"(p={comparison['psm_did']['did_results']['p_value']}, "
          f"{'显著' if comparison['psm_did']['did_results']['significant'] else '不显著'})")
    print(f"DML-ATE系数: {comparison['dml']['dml_results']['ate_coefficient']} "
          f"(p={comparison['dml']['dml_results']['p_value']}, "
          f"{'显著' if comparison['dml']['dml_results']['significant'] else '不显著'})")
    if comparison['dml']['dml_results']['significant']:
        print(f"DML 95%置信区间: [{comparison['dml']['dml_results']['ci_lower']}, "
              f"{comparison['dml']['dml_results']['ci_upper']}]")
    print(f"\n对比结论: {comparison['comparison']['conclusion']}")

    # 打印 LLM 解读（如果有）
    if pipeline.interpreter:
        print("\n" + "=" * 60)
        print("【普通DID的LLM解读】")
        print(comparison['naive_did']['llm_interpretation'])
        print("\n【PSM-DID的LLM解读】")
        print(comparison['psm_did']['llm_interpretation'])
        print("\n【DML的LLM解读】")
        print(comparison['dml']['llm_interpretation'])