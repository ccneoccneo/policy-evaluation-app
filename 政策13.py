"""
政策效果自动评估流水线（完整版：普通DID多规格、事件研究、PSM-DID、DML、中介效应、调节效应）
适配 Stata 数据格式（stkcd, year, DID, RES, DEPTH, EXTEND, 控制变量等）
修复事件研究中的数据类型错误
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


# ========== 数据准备模块（自动适配列名，支持 stkcd 和 DID） ==========
class PolicyDataPreparer:
    def __init__(self, data_path: str):
        # 支持 .dta 和 .csv
        if data_path.endswith('.dta'):
            self.df = pd.read_stata(data_path)
        else:
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

        # 实体列（支持 stkcd, id, 企业代码）
        if "stkcd" in self.df.columns:
            self.col_map["entity"] = "stkcd"
        elif "id" in self.df.columns:
            self.col_map["entity"] = "id"
        elif "企业代码" in self.df.columns:
            self.col_map["entity"] = "企业代码"
        else:
            raise ValueError("数据中缺少实体列（需要 'stkcd', 'id' 或 '企业代码'）")

        # 处理组列（支持 did, treat, DID）
        if "did" in self.df.columns:
            self.col_map["treat"] = "did"
        elif "treat" in self.df.columns:
            self.col_map["treat"] = "treat"
        elif "DID" in self.df.columns:
            self.col_map["treat"] = "DID"
        else:
            raise ValueError("数据中缺少处理组列（需要 'did', 'treat' 或 'DID'）")

        # 可选：城市列（用于城市固定效应）
        if "city" in self.df.columns:
            self.col_map["city"] = "city"
        else:
            self.col_map["city"] = None

        logger.info(
            f"列名映射: time={self.col_map['time']}, entity={self.col_map['entity']}, treat={self.col_map['treat']}, city={self.col_map['city']}")

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

        # 对连续变量取对数（可选）
        for col in [outcome_col] + control_cols:
            if col in df.columns and df[col].min() > 0:
                df[f"ln_{col}"] = np.log(df[col])
        return df

    def prepare_dml_data(self, outcome_col: str, treat_col: str,
                         feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备 DML 数据，支持面板去均值（固定效应）
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


# ========== 计量分析模块（增强：多固定效应、事件研究、中介、调节） ==========
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
                       entity_col=None, time_col=None, city_col=None) -> Dict:
        """
        支持企业、年份、城市固定效应的 DID 回归
        """
        try:
            from linearmodels.panel import PanelOLS
            # 构建固定效应
            fe_terms = []
            if entity_col and entity_col in df.columns:
                fe_terms.append("EntityEffects")
            if time_col and time_col in df.columns:
                fe_terms.append("TimeEffects")
            # 处理城市固定效应：如果提供了城市列，将其作为虚拟变量加入控制变量
            if city_col and city_col in df.columns:
                city_dummies = pd.get_dummies(df[city_col], prefix='city', drop_first=True)
                df = df.copy()
                df = pd.concat([df, city_dummies], axis=1)
                city_controls = list(city_dummies.columns)
                if controls:
                    controls = controls + city_controls
                else:
                    controls = city_controls
            # 设置面板索引
            df_panel = df.set_index([entity_col, time_col])
            ctrl_str = " + ".join(controls) if controls else ""
            fe_str = " + ".join(fe_terms)
            if fe_str:
                formula = f"{outcome} ~ {did_col} + {ctrl_str} + {fe_str}"
            else:
                formula = f"{outcome} ~ {did_col} + {ctrl_str}"
            model = PanelOLS.from_formula(formula, data=df_panel, drop_absorbed=True)
            res = model.fit(cov_type="clustered", cluster_entity=True)
        except Exception as e:
            logger.warning(f"PanelOLS 失败，降级为 OLS with HC1: {e}")
            # OLS 回归，支持城市固定效应
            formula = f"{outcome} ~ {did_col} + {treat_col} + {post_col}"
            if controls:
                formula += " + " + " + ".join(controls)
            if city_col and city_col in df.columns:
                city_dummies = pd.get_dummies(df[city_col], prefix='city', drop_first=True)
                df = pd.concat([df, city_dummies], axis=1)
                city_controls = list(city_dummies.columns)
                formula += " + " + " + ".join(city_controls)
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
    def event_study(df: pd.DataFrame, outcome: str, treat_col: str,
                    policy_year: int, time_col: str, leads: int = 5, lags: int = 7,
                    entity_col: str = None, controls: List[str] = None) -> Dict:
        """
        事件研究法（平行趋势检验），生成政策前后各期的系数和置信区间
        使用手动构建虚拟变量矩阵 + OLS，避免公式语法错误，同时支持高维固定效应（通过虚拟变量方式）
        """
        df = df.copy()
        df["relative_time"] = df[time_col] - policy_year
        # 构造相对时间虚拟变量，以 -1 期为基准
        time_dummies = {}
        for t in range(-leads, lags + 1):
            if t == -1:
                continue
            dummy_name = f"time_{t}"
            df[dummy_name] = (df["relative_time"] == t).astype(int)
            time_dummies[t] = dummy_name

        # 构建特征矩阵 X
        X = df[list(time_dummies.values())].copy()
        # 加入控制变量
        if controls:
            for c in controls:
                if c in df.columns:
                    # 确保控制变量为数值类型
                    X[c] = pd.to_numeric(df[c], errors='coerce')
        # 加入实体固定效应（stkcd）的虚拟变量（如果实体列存在）
        if entity_col and entity_col in df.columns:
            # 由于 stkcd 数量可能很多，这里使用 drop_first=True 避免完全共线性
            entity_dummies = pd.get_dummies(df[entity_col], prefix='ent', drop_first=True)
            # 确保虚拟变量为整数（0/1）
            entity_dummies = entity_dummies.astype(int)
            X = pd.concat([X, entity_dummies], axis=1)
        # 加入时间固定效应（year）的虚拟变量
        if time_col in df.columns:
            year_dummies = pd.get_dummies(df[time_col], prefix='yr', drop_first=True)
            year_dummies = year_dummies.astype(int)
            X = pd.concat([X, year_dummies], axis=1)
        # 添加常数项
        X = sm.add_constant(X)
        y = df[outcome]
        # 确保 y 为数值
        y = pd.to_numeric(y, errors='coerce')
        # 处理缺失值
        valid = ~(y.isna() | X.isna().any(axis=1))
        X = X[valid]
        y = y[valid]
        # 检查 X 的所有列是否为数值类型，如果不是则转换或删除
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col])
                except:
                    logger.warning(f"列 {col} 包含非数值数据，将被删除")
                    X = X.drop(columns=[col])
        # 检查是否有多重共线性导致奇异矩阵
        try:
            model = sm.OLS(y, X).fit(cov_type='HC1')
            coefs = {}
            ci_low = {}
            ci_high = {}
            for t, dummy in time_dummies.items():
                if dummy in model.params.index:
                    coefs[t] = model.params[dummy]
                    ci = model.conf_int().loc[dummy]
                    ci_low[t] = ci[0]
                    ci_high[t] = ci[1]
                else:
                    coefs[t] = np.nan
                    ci_low[t] = np.nan
                    ci_high[t] = np.nan
        except np.linalg.LinAlgError:
            logger.warning("设计矩阵奇异，尝试移除高共线性特征或使用更少的固定效应")
            # 回退方案：仅控制时间虚拟变量和控制变量，不加实体固定效应
            X_simple = df[list(time_dummies.values())].copy()
            if controls:
                for c in controls:
                    if c in df.columns:
                        X_simple[c] = pd.to_numeric(df[c], errors='coerce')
            if time_col in df.columns:
                year_dummies = pd.get_dummies(df[time_col], prefix='yr', drop_first=True).astype(int)
                X_simple = pd.concat([X_simple, year_dummies], axis=1)
            X_simple = sm.add_constant(X_simple)
            X_simple = X_simple[valid]
            # 再次确保数值类型
            for col in X_simple.columns:
                if X_simple[col].dtype == 'object':
                    X_simple = X_simple.drop(columns=[col])
            model_simple = sm.OLS(y, X_simple).fit(cov_type='HC1')
            coefs = {}
            ci_low = {}
            ci_high = {}
            for t, dummy in time_dummies.items():
                if dummy in model_simple.params.index:
                    coefs[t] = model_simple.params[dummy]
                    ci = model_simple.conf_int().loc[dummy]
                    ci_low[t] = ci[0]
                    ci_high[t] = ci[1]
                else:
                    coefs[t] = np.nan
                    ci_low[t] = np.nan
                    ci_high[t] = np.nan
        return {
            "coefficients": coefs,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "leads": leads,
            "lags": lags,
            "policy_year": policy_year
        }

    @staticmethod
    def plot_event_study(event_study_result: Dict, save_path="event_study.png"):
        """绘制事件研究图"""
        coefs = event_study_result["coefficients"]
        ci_low = event_study_result["ci_low"]
        ci_high = event_study_result["ci_high"]
        times = sorted([t for t in coefs.keys() if not np.isnan(coefs[t])])
        if not times:
            logger.warning("没有可用的系数，无法绘制事件研究图")
            return
        coef_vals = [coefs[t] for t in times]
        ci_low_vals = [ci_low[t] for t in times]
        ci_high_vals = [ci_high[t] for t in times]
        plt.figure(figsize=(12, 6))
        plt.errorbar(times, coef_vals, yerr=[(coef_vals[i] - ci_low_vals[i]) for i in range(len(times))],
                     fmt='o', capsize=5, color='steelblue', ecolor='gray', elinewidth=2)
        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='零线')
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, label='政策实施时点')
        plt.xlabel("相对政策时点", fontsize=12)
        plt.ylabel("系数估计值", fontsize=12)
        plt.title("事件研究图（平行趋势检验）", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"事件研究图已保存至 {save_path}")

    @staticmethod
    def mediation_analysis(df: pd.DataFrame, treatment: str, mediator: str, outcome: str,
                           controls: List[str], fixed_effects: List[str] = None) -> Dict:
        """
        中介效应三步法（Baron & Kenny），并计算 Sobel 检验
        """
        # 构造固定效应虚拟变量
        if fixed_effects:
            fe_terms = " + ".join([f"C({fe})" for fe in fixed_effects])
        else:
            fe_terms = ""
        # 第一步：treatment -> mediator
        formula1 = f"{mediator} ~ {treatment} + " + " + ".join(controls)
        if fe_terms:
            formula1 += " + " + fe_terms
        res1 = ols(formula1, data=df).fit(cov_type="HC1")
        a = res1.params[treatment]
        a_se = res1.bse[treatment]
        a_p = res1.pvalues[treatment]

        # 第二步：treatment -> outcome
        formula2 = f"{outcome} ~ {treatment} + " + " + ".join(controls)
        if fe_terms:
            formula2 += " + " + fe_terms
        res2 = ols(formula2, data=df).fit(cov_type="HC1")
        c = res2.params[treatment]
        c_se = res2.bse[treatment]
        c_p = res2.pvalues[treatment]

        # 第三步：treatment + mediator -> outcome
        formula3 = f"{outcome} ~ {treatment} + {mediator} + " + " + ".join(controls)
        if fe_terms:
            formula3 += " + " + fe_terms
        res3 = ols(formula3, data=df).fit(cov_type="HC1")
        c_prime = res3.params[treatment]
        c_prime_se = res3.bse[treatment]
        c_prime_p = res3.pvalues[treatment]
        b = res3.params[mediator]
        b_se = res3.bse[mediator]
        b_p = res3.pvalues[mediator]

        # 间接效应 = a * b，Sobel 检验
        sobel_se = np.sqrt(b**2 * a_se**2 + a**2 * b_se**2)
        indirect = a * b
        sobel_z = indirect / sobel_se
        from scipy import stats
        sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

        return {
            "a_coef": a, "a_p": a_p,
            "c_coef": c, "c_p": c_p,
            "b_coef": b, "b_p": b_p,
            "c_prime_coef": c_prime, "c_prime_p": c_prime_p,
            "indirect_effect": indirect,
            "sobel_z": sobel_z,
            "sobel_p": sobel_p,
            "mediation_ratio": indirect / c if c != 0 else np.nan
        }

    @staticmethod
    def interaction_analysis(df: pd.DataFrame, treatment: str, moderator: str, outcome: str,
                             controls: List[str], fixed_effects: List[str] = None) -> Dict:
        """
        调节效应分析（交互项）
        """
        df = df.copy()
        interaction = f"{treatment}_X_{moderator}"
        df[interaction] = df[treatment] * df[moderator]
        formula = f"{outcome} ~ {treatment} + {moderator} + {interaction} + " + " + ".join(controls)
        if fixed_effects:
            formula += " + " + " + ".join([f"C({fe})" for fe in fixed_effects])
        res = ols(formula, data=df).fit(cov_type="HC1")
        coef_interaction = res.params[interaction]
        p_interaction = res.pvalues[interaction]
        return {
            "interaction_term": interaction,
            "coefficient": coef_interaction,
            "p_value": p_interaction,
            "significant": p_interaction < 0.05,
            "full_results": res.summary()
        }


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
        if parallel_trend_result and "coefficients" in parallel_trend_result:
            coefs = parallel_trend_result["coefficients"]
            ci_low = parallel_trend_result["ci_low"]
            ci_high = parallel_trend_result["ci_high"]
            pre_times = [t for t in coefs.keys() if t < 0]
            any_sig = any(ci_low[t] > 0 or ci_high[t] < 0 for t in pre_times if not np.isnan(ci_low[t]))
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


# ========== 流水线整合（完整版） ==========
class PolicyEvaluationPipeline:
    def __init__(self, data_path: str, llm_engine=None):
        self.preparer = PolicyDataPreparer(data_path)
        self.analyzer = CausalInferenceAnalyzer()
        self.dml_analyzer = DoubleMachineLearningAnalyzer()
        self.interpreter = PolicyEvalInterpreter(llm_engine) if llm_engine else None

    def run_naive_did_with_multiple_specs(self, policy_name: str, policy_year: int,
                                          treat_col: str, outcome_col: str,
                                          control_cols: List[str],
                                          fixed_effects_list: List[List[str]]) -> List[Dict]:
        """
        运行多个固定效应规格的普通 DID，返回结果列表
        fixed_effects_list: 例如 [['stkcd'], ['stkcd','year'], ['stkcd','year','city']]
        """
        results = []
        actual_time_col = self.preparer.col_map["time"]
        actual_entity_col = self.preparer.col_map["entity"]
        actual_city_col = self.preparer.col_map.get("city", None)
        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)
        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df.columns else outcome_col
        ctrl_ln = [f"ln_{c}" if f"ln_{c}" in df.columns else c for c in control_cols]

        for fe_spec in fixed_effects_list:
            entity_col = actual_entity_col if 'stkcd' in fe_spec or actual_entity_col in fe_spec else None
            time_col = actual_time_col if 'year' in fe_spec else None
            city_col = actual_city_col if 'city' in fe_spec else None
            did_res = self.analyzer.did_regression(
                df, outcome_ln,
                did_col="did_interaction",
                treat_col=treat_col,
                post_col="post",
                controls=ctrl_ln,
                entity_col=entity_col,
                time_col=time_col,
                city_col=city_col
            )
            results.append({
                "fixed_effects": fe_spec,
                "did_coefficient": did_res["did_coefficient"],
                "p_value": did_res["p_value"],
                "significant": did_res["significant"],
                "r_squared": did_res["r_squared"],
                "n_obs": did_res["n_obs"]
            })
        return results

    def run_naive_did_with_event_study(self, policy_name: str, policy_year: int,
                                       treat_col: str, outcome_col: str,
                                       control_cols: List[str]) -> Dict:
        """
        普通 DID 并绘制事件研究图
        """
        actual_time_col = self.preparer.col_map["time"]
        actual_entity_col = self.preparer.col_map["entity"]
        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)
        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df.columns else outcome_col
        # 事件研究
        event_result = self.analyzer.event_study(
            df, outcome_ln, treat_col, policy_year, actual_time_col,
            leads=5, lags=7, entity_col=actual_entity_col, controls=control_cols
        )
        if event_result:
            self.analyzer.plot_event_study(event_result, f"event_study_naive_{policy_name}.png")
        # DID 回归（包含企业、年份固定效应）
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
        interpretation = ""
        if self.interpreter:
            interpretation = self.interpreter.interpret_did(
                "普通DID(事件研究)", did_res, policy_name, outcome_col, event_result, policy_year
            )
        return {
            "did_results": did_res,
            "event_study": event_result,
            "llm_interpretation": interpretation
        }

    def run_psm_did_with_event_study(self, policy_name: str, policy_year: int,
                                     treat_col: str, outcome_col: str,
                                     control_cols: List[str], match_vars: List[str]) -> Dict:
        """
        PSM-DID 并绘制事件研究图（匹配后样本）
        """
        actual_time_col = self.preparer.col_map["time"]
        actual_entity_col = self.preparer.col_map["entity"]
        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)
        df_matched = self.analyzer.psm_matching(df, treat_col, match_vars)
        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df_matched.columns else outcome_col
        # 事件研究（匹配后）
        event_result = self.analyzer.event_study(
            df_matched, outcome_ln, treat_col, policy_year, actual_time_col,
            leads=5, lags=7, entity_col=actual_entity_col, controls=control_cols
        )
        if event_result:
            self.analyzer.plot_event_study(event_result, f"event_study_psm_{policy_name}.png")
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
        interpretation = ""
        if self.interpreter:
            interpretation = self.interpreter.interpret_did(
                "PSM-DID(事件研究)", did_res, policy_name, outcome_col, event_result, policy_year
            )
        return {
            "did_results": did_res,
            "event_study": event_result,
            "llm_interpretation": interpretation
        }

    def run_dml(self, policy_name: str, outcome_col: str, treat_col: str,
                feature_cols: List[str]) -> Dict:
        """双重机器学习（DML）"""
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

    def run_mediation(self, treatment: str, mediator: str, outcome: str,
                      controls: List[str], fixed_effects: List[str] = None) -> Dict:
        """中介效应分析"""
        df = self.preparer.df.copy()
        # 确保有 did_interaction 列（如果使用原始 DID 变量，需调整）
        if "did_interaction" not in df.columns:
            # 简化：假设传入的 treatment 是原始处理变量（如 DID），需先生成交互项
            # 但这里为了通用，要求用户传入已经构造好的处理变量（如 did_interaction）
            # 实际使用时，建议在调用前构造好。
            pass
        return self.analyzer.mediation_analysis(df, treatment, mediator, outcome, controls, fixed_effects)

    def run_interaction(self, treatment: str, moderator: str, outcome: str,
                        controls: List[str], fixed_effects: List[str] = None) -> Dict:
        """调节效应分析（交互项）"""
        df = self.preparer.df.copy()
        return self.analyzer.interaction_analysis(df, treatment, moderator, outcome, controls, fixed_effects)

    def run_full_comparison(self, policy_name: str, policy_year: int,
                            treat_col: str, outcome_col: str,
                            control_cols: List[str], match_vars: List[str],
                            dml_features: List[str]) -> Dict:
        """
        完整对比：普通DID（事件研究）、PSM-DID（事件研究）、DML
        """
        logger.info(f"开始三种方法对比评估：{policy_name}")

        # 普通DID with 事件研究
        naive_result = self.run_naive_did_with_event_study(
            policy_name, policy_year, treat_col, outcome_col, control_cols
        )

        # PSM-DID with 事件研究
        psm_result = self.run_psm_did_with_event_study(
            policy_name, policy_year, treat_col, outcome_col, control_cols, match_vars
        )

        # DML
        dml_result = self.run_dml(policy_name, outcome_col, treat_col, dml_features)

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
# ========== 主程序 ==========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 数据路径
    DATA_PATH = "附件2 智能制造政策、知识重构与供应链韧性.dta"

    # 初始化通义千问
    import os
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-2f95ab436b644f11849c067a74744c7a")
    llm_engine = QwenEngine(api_key=api_key, model="qwen-turbo")

    pipeline = PolicyEvaluationPipeline(DATA_PATH, llm_engine=llm_engine)

    # ========== 参数设置 ==========
    POLICY_NAME = "智能制造政策"
    POLICY_YEAR = 2015
    TREAT_COL = "DID"
    OUTCOME_COL = "RES"
    CONTROL_COLS = ["CRE", "IA", "UD", "AGE", "AGE2", "SIZE", "PROFIT",
                    "TOP5", "BOARD", "RD", "FAG", "LE", "OPEN", "GOV"]
    MATCH_VARS = CONTROL_COLS
    DML_FEATURES = CONTROL_COLS
    # ============================

    # 1. 普通DID多规格回归
    print("\n=== 普通DID多规格回归 ===")
    fe_specs = [[], ['stkcd'], ['stkcd', 'year'], ['stkcd', 'year', 'city']]
    multi_results = pipeline.run_naive_did_with_multiple_specs(
        POLICY_NAME, POLICY_YEAR, TREAT_COL, OUTCOME_COL, CONTROL_COLS, fe_specs
    )
    for i, res in enumerate(multi_results):
        print(f"规格{i+1} {res['fixed_effects']}: DID系数={res['did_coefficient']}, p={res['p_value']}, 显著={res['significant']}")

    # 2. 普通DID + 事件研究图（如果出错可跳过，不影响后续）
    print("\n=== 普通DID + 事件研究图 ===")
    try:
        naive_event = pipeline.run_naive_did_with_event_study(
            POLICY_NAME, POLICY_YEAR, TREAT_COL, OUTCOME_COL, CONTROL_COLS
        )
        print(f"DID系数: {naive_event['did_results']['did_coefficient']}, p={naive_event['did_results']['p_value']}")
        if naive_event['llm_interpretation']:
            print(f"普通DID的LLM解读:\n{naive_event['llm_interpretation']}")
    except Exception as e:
        print(f"普通DID事件研究图失败: {e}")

    # 3. PSM-DID + 事件研究图
    print("\n=== PSM-DID + 事件研究图 ===")
    try:
        psm_event = pipeline.run_psm_did_with_event_study(
            POLICY_NAME, POLICY_YEAR, TREAT_COL, OUTCOME_COL, CONTROL_COLS, MATCH_VARS
        )
        print(f"DID系数: {psm_event['did_results']['did_coefficient']}, p={psm_event['did_results']['p_value']}")
        if psm_event['llm_interpretation']:
            print(f"PSM-DID的LLM解读:\n{psm_event['llm_interpretation']}")
    except Exception as e:
        print(f"PSM-DID事件研究图失败: {e}")

    # 4. DML 评估（单独输出）
    print("\n=== 双重机器学习（DML）评估 ===")
    dml_result = pipeline.run_dml(POLICY_NAME, OUTCOME_COL, TREAT_COL, DML_FEATURES)
    print(f"DML-ATE系数: {dml_result['dml_results']['ate_coefficient']}")
    print(f"标准误: {dml_result['dml_results']['std_error']}")
    print(f"95%置信区间: [{dml_result['dml_results']['ci_lower']}, {dml_result['dml_results']['ci_upper']}]")
    print(f"是否显著: {dml_result['dml_results']['significant']}")
    if dml_result['llm_interpretation']:
        print(f"DML的LLM解读:\n{dml_result['llm_interpretation']}")

    # 5. 三种方法完整对比（包含综合LLM解读）
    print("\n=== 三种方法完整对比 ===")
    comparison = pipeline.run_full_comparison(
        POLICY_NAME, POLICY_YEAR, TREAT_COL, OUTCOME_COL,
        CONTROL_COLS, MATCH_VARS, DML_FEATURES
    )
    print(f"普通DID系数: {comparison['comparison']['naive_did']['coefficient']}")
    print(f"PSM-DID系数: {comparison['comparison']['psm_did']['coefficient']}")
    print(f"DML系数: {comparison['comparison']['dml']['coefficient']}")
    print(f"对比结论: {comparison['comparison']['conclusion']}")

    # 6. 生成三种方法的综合LLM解读（最终报告）
    if pipeline.interpreter:
        summary_prompt = f"""
你是专业的政策评估研究员。请基于以下三种方法对【{POLICY_NAME}】的评估结果，撰写一份综合政策解读报告，要求：
1. 对比普通DID、PSM-DID、DML三种方法的估计结果，说明它们的一致性和差异。
2. 分析可能产生差异的原因（如选择性偏差、遗漏变量、非线性关系等）。
3. 给出最终的政策结论和三条具体建议。

评估结果：
- 普通DID: 系数={comparison['comparison']['naive_did']['coefficient']}, p={comparison['comparison']['naive_did']['p_value']}, {'显著' if comparison['comparison']['naive_did']['significant'] else '不显著'}
- PSM-DID: 系数={comparison['comparison']['psm_did']['coefficient']}, p={comparison['comparison']['psm_did']['p_value']}, {'显著' if comparison['comparison']['psm_did']['significant'] else '不显著'}
- DML: ATE={comparison['comparison']['dml']['coefficient']}, p={comparison['comparison']['dml']['p_value']}, {'显著' if comparison['comparison']['dml']['significant'] else '不显著'}, 95%CI=[{comparison['comparison']['dml']['ci_lower']}, {comparison['comparison']['dml']['ci_upper']}]

输出500-800字，分三部分：方法对比、原因分析、政策建议。
"""
        final_interpretation = pipeline.interpreter.llm.generate("政策评估专家", summary_prompt)
        print("\n=== 三种方法综合LLM解读 ===")
        print(final_interpretation)