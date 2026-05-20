import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.formula.api import ols
from typing import Dict, List, Tuple, Optional
import dashscope
from http import HTTPStatus
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy import stats

st.set_page_config(page_title="政策效果评估系统", layout="wide")
st.title("📊 政策效果自动评估系统")
st.markdown("支持普通DID、PSM-DID、双重机器学习（DML）、事件研究法（Event Study），集成通义千问LLM解读")

# ---------- 初始化 session ----------
if "results" not in st.session_state:
    st.session_state.results = {}
if "df" not in st.session_state:
    st.session_state.df = None

run_button = False


# ========== 通义千问引擎 ==========
class QwenEngine:
    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        if not api_key:
            raise ValueError("API Key 不能为空")
        self.api_key = api_key
        self.model = model
        dashscope.api_key = api_key

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 1000, temperature: float = 0.4) -> str:
        full_prompt = f"系统指令：{system_prompt}\n\n用户问题：{user_prompt}"
        return self._safe_api_call(full_prompt, max_tokens, temperature)

    def _safe_api_call(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个专业的政策评估研究员，熟悉计量经济学、因果推断、"
                        "管理学和经济学实证论文写作。请用中文回答，并优先采用严谨、克制、"
                        "符合学术规范的表达。"
                    )
                },
                {"role": "user", "content": prompt}
            ]
            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                result_format="message"
            )

            if response.status_code != HTTPStatus.OK:
                return f"API 调用失败: {response.code} - {response.message}"

            if (hasattr(response, "output") and response.output and
                hasattr(response.output, "choices") and response.output.choices and
                len(response.output.choices) > 0):
                choice = response.output.choices[0]
                if hasattr(choice, "message") and choice.message and hasattr(choice.message, "content"):
                    content = choice.message.content
                    if content:
                        return content

            if hasattr(response, "output") and hasattr(response.output, "text"):
                return response.output.text

            return "API 返回数据格式异常，无法提取内容。"

        except Exception as e:
            return f"系统错误: {str(e)}"


# ========== 数据准备模块 ==========
class PolicyDataPreparer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = [col.strip() for col in self.df.columns]
        self.col_map = {}

        if "year" in self.df.columns:
            self.col_map["time"] = "year"
        elif "年份" in self.df.columns:
            self.col_map["time"] = "年份"
        else:
            raise ValueError("数据中缺少时间列（year/年份）")

        if "stkcd" in self.df.columns:
            self.col_map["entity"] = "stkcd"
        elif "id" in self.df.columns:
            self.col_map["entity"] = "id"
        elif "企业代码" in self.df.columns:
            self.col_map["entity"] = "企业代码"
        else:
            raise ValueError("数据中缺少实体列（stkcd/id/企业代码）")

        if "city" in self.df.columns:
            self.col_map["city"] = "city"
        elif "城市" in self.df.columns:
            self.col_map["city"] = "城市"
        else:
            self.col_map["city"] = None

    def prepare_did_data(self, policy_year: int, treat_col: str,
                         outcome_col: str, control_cols: List[str]) -> pd.DataFrame:
        df = self.df.copy()
        actual_time_col = self.col_map["time"]

        df[actual_time_col] = pd.to_numeric(df[actual_time_col], errors="coerce")
        df[treat_col] = pd.to_numeric(df[treat_col], errors="coerce")
        df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")

        df["post"] = (df[actual_time_col] >= policy_year).astype(int)
        df["did_interaction"] = df[treat_col] * df["post"]

        for col in [outcome_col] + control_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                if series.notna().any() and series.min() > 0:
                    df[f"ln_{col}"] = np.log(series)

        return df

    def prepare_dml_data(self, outcome_col: str, treat_col: str,
                         feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = self.df.copy()
        actual_entity_col = self.col_map["entity"]

        for col in [outcome_col, treat_col] + feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[f"{col}_demeaned"] = df.groupby(actual_entity_col)[col].transform(lambda x: x - x.mean())

        y_col = f"{outcome_col}_demeaned"
        t_col = f"{treat_col}_demeaned"
        x_cols = [f"{c}_demeaned" for c in feature_cols if f"{c}_demeaned" in df.columns]

        if y_col not in df.columns or t_col not in df.columns or len(x_cols) == 0:
            raise ValueError("DML 所需变量准备失败，请检查结果变量、处理变量和特征变量是否为数值型且存在。")

        Y = df[y_col].values
        T = df[t_col].values
        X = df[x_cols].values

        mask = ~(np.isnan(Y) | np.isnan(T) | np.isnan(X).any(axis=1))
        Y, T, X = Y[mask], T[mask], X[mask]

        if len(Y) == 0:
            raise ValueError("DML 清洗后无可用样本，请检查变量缺失情况。")

        X = StandardScaler().fit_transform(X)
        return Y, T, X

    def prepare_event_study_data(self, policy_year: int, treat_col: str,
                                 window_pre: int, window_post: int) -> pd.DataFrame:
        df = self.df.copy()
        time_col = self.col_map["time"]

        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df[treat_col] = pd.to_numeric(df[treat_col], errors="coerce")

        df["event_time"] = df[time_col] - policy_year
        df["event_time"] = df["event_time"].clip(lower=-window_pre, upper=window_post)

        for k in range(-window_pre, window_post + 1):
            if k == -1:
                continue
            df[f"event_{k}"] = ((df["event_time"] == k) & (df[treat_col] == 1)).astype(int)

        return df


# ========== 计量分析模块 ==========
class CausalInferenceAnalyzer:
    @staticmethod
    def psm_matching(df: pd.DataFrame, treat_col: str, match_vars: List[str], caliper: float = 0.05) -> pd.DataFrame:
        if len(match_vars) == 0:
            raise ValueError("PSM 匹配变量不能为空。")

        X = df[match_vars].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        y = pd.to_numeric(df[treat_col], errors="coerce")

        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        df_valid = df.loc[valid_mask].copy()

        if y.nunique() < 2:
            raise ValueError("处理组变量只有一个取值，无法进行 PSM 匹配。")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logit = LogisticRegression(max_iter=1000)
        logit.fit(X_scaled, y)

        df_valid["pscore"] = logit.predict_proba(X_scaled)[:, 1]
        treat_df = df_valid[df_valid[treat_col] == 1].copy()
        ctrl_df = df_valid[df_valid[treat_col] == 0].copy()

        if treat_df.empty or ctrl_df.empty:
            raise ValueError("处理组或控制组为空，无法进行 PSM 匹配。")

        matched_ctrl_ids = []
        used_ctrl_ids = set()

        for _, t_row in treat_df.iterrows():
            candidate_ctrl = ctrl_df.loc[~ctrl_df.index.isin(used_ctrl_ids)].copy()
            if candidate_ctrl.empty:
                break

            diffs = (candidate_ctrl["pscore"] - t_row["pscore"]).abs()
            min_diff = diffs.min()

            if min_diff <= caliper:
                best_idx = diffs.idxmin()
                matched_ctrl_ids.append(best_idx)
                used_ctrl_ids.add(best_idx)

        if len(matched_ctrl_ids) == 0:
            raise ValueError("未找到满足 caliper 条件的匹配样本，请放宽 caliper 或调整匹配变量。")

        matched_ctrl = ctrl_df.loc[matched_ctrl_ids]
        matched_df = pd.concat([treat_df, matched_ctrl], axis=0).copy()
        return matched_df

    @staticmethod
    def did_regression(df: pd.DataFrame, outcome: str,
                       did_col: str = "did_interaction", treat_col: str = "treat",
                       post_col: str = "post", controls: Optional[List[str]] = None,
                       entity_col: Optional[str] = None, time_col: Optional[str] = None,
                       city_col: Optional[str] = None) -> Dict:
        controls = controls or []
        working_df = df.copy()

        try:
            from linearmodels.panel import PanelOLS

            fe_terms = []

            if city_col and city_col in working_df.columns:
                city_dummies = pd.get_dummies(working_df[city_col], prefix="city", drop_first=True)
                working_df = pd.concat([working_df, city_dummies], axis=1)
                controls = controls + list(city_dummies.columns)

            if entity_col and entity_col in working_df.columns:
                fe_terms.append("EntityEffects")
            if time_col and time_col in working_df.columns:
                fe_terms.append("TimeEffects")

            if entity_col and time_col:
                df_panel = working_df.set_index([entity_col, time_col])

                rhs_terms = [did_col]
                if len(controls) > 0:
                    rhs_terms.extend(controls)
                if len(fe_terms) > 0:
                    rhs_terms.extend(fe_terms)

                formula = f"{outcome} ~ {' + '.join(rhs_terms)}"
                model = PanelOLS.from_formula(formula, data=df_panel, drop_absorbed=True)
                res = model.fit(cov_type="clustered", cluster_entity=True)

                did_coef = res.params.get(did_col, np.nan)
                did_pval = res.pvalues.get(did_col, np.nan)

                r_squared_val = np.nan
                try:
                    r_squared_val = float(res.rsquared)
                except Exception:
                    try:
                        r_squared_val = float(res.rsquared_overall)
                    except Exception:
                        r_squared_val = np.nan

                return {
                    "did_coefficient": round(float(did_coef), 4) if pd.notna(did_coef) else np.nan,
                    "p_value": round(float(did_pval), 4) if pd.notna(did_pval) else np.nan,
                    "significant": bool(pd.notna(did_pval) and did_pval < 0.05),
                    "r_squared": round(float(r_squared_val), 4) if pd.notna(r_squared_val) else np.nan,
                    "n_obs": int(res.nobs),
                    "model_summary": str(res.summary),
                    "model_type": "PanelOLS"
                }

        except Exception:
            pass

        formula = f"{outcome} ~ {did_col} + {treat_col} + {post_col}"
        if len(controls) > 0:
            formula += " + " + " + ".join(controls)

        if city_col and city_col in working_df.columns:
            city_dummies = pd.get_dummies(working_df[city_col], prefix="city", drop_first=True)
            working_df = pd.concat([working_df, city_dummies], axis=1)
            city_controls = list(city_dummies.columns)
            if len(city_controls) > 0:
                formula += " + " + " + ".join(city_controls)

        if entity_col and entity_col in working_df.columns:
            formula += f" + C({entity_col})"
        if time_col and time_col in working_df.columns:
            formula += f" + C({time_col})"

        res = ols(formula, data=working_df).fit(cov_type="HC1")

        did_coef = res.params.get(did_col, np.nan)
        did_pval = res.pvalues.get(did_col, np.nan)

        return {
            "did_coefficient": round(float(did_coef), 4) if pd.notna(did_coef) else np.nan,
            "p_value": round(float(did_pval), 4) if pd.notna(did_pval) else np.nan,
            "significant": bool(pd.notna(did_pval) and did_pval < 0.05),
            "r_squared": round(float(res.rsquared), 4) if pd.notna(res.rsquared) else np.nan,
            "n_obs": int(res.nobs),
            "model_summary": str(res.summary()),
            "model_type": "OLS"
        }

    @staticmethod
    def event_study_regression(df: pd.DataFrame, outcome: str,
                               controls: Optional[List[str]] = None,
                               entity_col: Optional[str] = None,
                               time_col: Optional[str] = None,
                               window_pre: int = 4,
                               window_post: int = 4) -> Dict:
        controls = controls or []
        working_df = df.copy()

        event_vars = []
        for k in range(-window_pre, window_post + 1):
            if k == -1:
                continue
            col_name = f"event_{k}"
            if col_name in working_df.columns:
                event_vars.append(col_name)

        formula = f"{outcome} ~ " + " + ".join(event_vars)
        if len(controls) > 0:
            formula += " + " + " + ".join(controls)

        if entity_col and entity_col in working_df.columns:
            formula += f" + C({entity_col})"
        if time_col and time_col in working_df.columns:
            formula += f" + C({time_col})"

        res = ols(formula, data=working_df).fit(cov_type="HC1")

        rows = []
        for k in range(-window_pre, window_post + 1):
            if k == -1:
                continue
            name = f"event_{k}"
            coef = res.params.get(name, np.nan)
            se = res.bse.get(name, np.nan)
            p = res.pvalues.get(name, np.nan)
            rows.append({
                "event_time": k,
                "coef": coef,
                "std_error": se,
                "p_value": p,
                "ci_lower": coef - 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan,
                "ci_upper": coef + 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan
            })

        return {
            "event_results": pd.DataFrame(rows),
            "model_summary": str(res.summary())
        }


# ========== 双重机器学习模块 ==========
class DoubleMachineLearningAnalyzer:
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self._fitted = False
        self._ate = None
        self._ate_std = None
        self._ate_pvalue = None
        self._ate_ci = None

    def _get_default_models(self):
        model_y = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=10,
            random_state=42
        )
        model_t = RidgeCV(alphas=[0.1, 1.0, 10.0])
        return model_y, model_t

    def _t_stat_pvalue(self, t_value, df=100):
        return 2 * (1 - stats.t.cdf(abs(t_value), df=df))

    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray):
        if len(Y) < self.n_folds:
            raise ValueError("DML 样本量小于折数，无法进行交叉拟合。")

        model_y, model_t = self._get_default_models()
        n = len(Y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        Y_res = np.zeros(n)
        T_res = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            T_train, T_val = T[train_idx], T[val_idx]

            model_y.fit(X_train, Y_train)
            model_t.fit(X_train, T_train)

            Y_pred = model_y.predict(X_val)
            T_pred = model_t.predict(X_val)

            Y_res[val_idx] = Y_val - Y_pred
            T_res[val_idx] = T_val - T_pred

        denom = np.sum(T_res ** 2)
        if denom < 1e-10:
            T_res = T_res + np.random.normal(0, 1e-8, size=n)
            denom = np.sum(T_res ** 2)

        theta = np.sum(Y_res * T_res) / denom
        resid = Y_res - theta * T_res
        sigma2 = np.mean(resid ** 2)
        var_theta = sigma2 / denom
        std_theta = np.sqrt(var_theta)

        self._ate = theta
        self._ate_std = std_theta

        df_approx = max(10, n - X.shape[1])
        t_stat = theta / std_theta if std_theta > 0 else np.nan
        self._ate_pvalue = self._t_stat_pvalue(t_stat, df=df_approx) if pd.notna(t_stat) else np.nan
        self._ate_ci = (theta - 1.96 * std_theta, theta + 1.96 * std_theta)

        self._fitted = True
        return self

    def get_ate_results(self) -> Dict:
        if not self._fitted:
            raise ValueError("模型未拟合")

        return {
            "ate_coefficient": round(float(self._ate), 4),
            "std_error": round(float(self._ate_std), 4),
            "p_value": round(float(self._ate_pvalue), 4) if pd.notna(self._ate_pvalue) else np.nan,
            "significant": bool(pd.notna(self._ate_pvalue) and self._ate_pvalue < 0.05),
            "ci_lower": round(float(self._ate_ci[0]), 4),
            "ci_upper": round(float(self._ate_ci[1]), 4),
            "method": "Double Machine Learning (DML)",
        }


# ========== 评估流水线 ==========
class PolicyEvaluationPipeline:
    def __init__(self, df: pd.DataFrame, llm_engine=None, llm_style: str = "论文摘要风格",
                 research_background: str = "", mechanism_hint: str = "", heterogeneity_hint: str = ""):
        self.preparer = PolicyDataPreparer(df)
        self.analyzer = CausalInferenceAnalyzer()
        self.dml_analyzer = DoubleMachineLearningAnalyzer()
        self.llm = llm_engine
        self.llm_style = llm_style
        self.research_background = research_background
        self.mechanism_hint = mechanism_hint
        self.heterogeneity_hint = heterogeneity_hint

    def _format_sig_text(self, p_value: float, significant: bool) -> str:
        if pd.isna(p_value):
            return "统计显著性未知"
        return "在统计上显著" if significant else "在统计上不显著"

    def _build_abstract_style_prompt(
        self,
        method_name: str,
        policy_name: str,
        policy_year: int,
        outcome_col: str,
        coef: float,
        p_value: float,
        significant: bool,
        n_obs: Optional[int] = None,
        r_squared: Optional[float] = None,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None
    ) -> str:
        sig_text = self._format_sig_text(p_value, significant)

        ci_text = ""
        if ci_lower is not None and ci_upper is not None:
            ci_text = f"95%置信区间为[{ci_lower}, {ci_upper}]。"

        sample_text = ""
        if n_obs is not None:
            sample_text += f"样本量为{n_obs}。"
        if r_squared is not None and pd.notna(r_squared):
            sample_text += f"模型拟合优度R²为{r_squared}。"

        background_text = (
            self.research_background
            if self.research_background
            else "面对高质量发展和产业转型升级要求，相关政策已成为推动企业能力提升与绩效改善的重要制度安排。"
        )
        mechanism_text = (
            self.mechanism_hint
            if self.mechanism_hint
            else "技术升级、资源配置优化、知识重构、组织协同提升"
        )
        heterogeneity_text = (
            self.heterogeneity_hint
            if self.heterogeneity_hint
            else "政策执行能力、地区制度环境、企业治理水平、供需关系稳定性"
        )

        return f"""
请模仿中文CSSCI/经管类学术论文摘要的写法，围绕政策评估结果生成一段“摘要式解读”。

写作要求：
1. 输出为一个完整自然段，不要分点，不要加标题。
2. 风格正式、凝练、学术化，尽量接近论文摘要。
3. 结构应尽量包含：研究背景、样本与方法、核心结果、机制解释、异质性或稳健性表述、结论含义。
4. 若当前只提供了单一模型结果，机制、异质性、稳健性部分必须使用审慎表述，如“可能表明”“后续可进一步检验”“有待进一步识别”等。
5. 不得虚构未提供的具体检验结果；尤其不要把尚未完成的事件研究、机制检验、异质性检验、溢出效应检验写成既成事实。
6. 输出字数控制在300-500字。
7. 全部使用中文。

已知信息：
- 研究背景：{background_text}
- 政策名称：{policy_name}
- 政策实施年份：{policy_year}
- 结果变量：{outcome_col}
- 估计方法：{method_name}
- 核心系数：{coef}
- P值：{p_value}，即该结果{sig_text}
- {sample_text}
- {ci_text}
- 可参考的潜在作用机制：{mechanism_text}
- 可参考的异质性维度：{heterogeneity_text}

请直接输出符合论文摘要风格的结果解读正文。
"""

    def _build_report_style_prompt(
        self,
        method_name: str,
        policy_name: str,
        outcome_col: str,
        coef: float,
        p_value: float,
        significant: bool,
        n_obs: Optional[int] = None,
        r_squared: Optional[float] = None,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None
    ) -> str:
        extra = ""
        if ci_lower is not None and ci_upper is not None:
            extra += f"\n95%置信区间：[{ci_lower}, {ci_upper}]"
        if r_squared is not None and pd.notna(r_squared):
            extra += f"\nR²：{r_squared}"
        if n_obs is not None:
            extra += f"\n样本量：{n_obs}"

        return f"""
你是专业的政策评估研究员。请基于{method_name}方法的估计结果，撰写政府研究报告风格的政策解读，分三段：
1. 政策效果总结（方向、大小、显著性）
2. 可能的影响机制或解释
3. 三条具体政策建议

政策名称：{policy_name}
结果变量：{outcome_col}
核心系数：{coef}
P值：{p_value}（{'显著' if significant else '不显著'}）
{extra}

输出300-500字。
"""

    def _generate_single_method_interpretation(
        self,
        method_name: str,
        policy_name: str,
        policy_year: int,
        outcome_col: str,
        coef: float,
        p_value: float,
        significant: bool,
        n_obs: Optional[int] = None,
        r_squared: Optional[float] = None,
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None
    ) -> str:
        if not self.llm:
            return ""

        if self.llm_style == "论文摘要风格":
            prompt = self._build_abstract_style_prompt(
                method_name=method_name,
                policy_name=policy_name,
                policy_year=policy_year,
                outcome_col=outcome_col,
                coef=coef,
                p_value=p_value,
                significant=significant,
                n_obs=n_obs,
                r_squared=r_squared,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            )
        else:
            prompt = self._build_report_style_prompt(
                method_name=method_name,
                policy_name=policy_name,
                outcome_col=outcome_col,
                coef=coef,
                p_value=p_value,
                significant=significant,
                n_obs=n_obs,
                r_squared=r_squared,
                ci_lower=ci_lower,
                ci_upper=ci_upper
            )

        return self.llm.generate("政策评估专家", prompt)

    def run_naive_did_with_multiple_specs(self, policy_year: int, treat_col: str,
                                          outcome_col: str, control_cols: List[str],
                                          fixed_effects_list: List[List[str]]) -> List[Dict]:
        results = []
        actual_time_col = self.preparer.col_map["time"]
        actual_entity_col = self.preparer.col_map["entity"]
        actual_city_col = self.preparer.col_map.get("city", None)

        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)
        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df.columns else outcome_col
        ctrl_ln = [f"ln_{c}" if f"ln_{c}" in df.columns else c for c in control_cols]

        for fe_spec in fixed_effects_list:
            entity_col = actual_entity_col if actual_entity_col in fe_spec else None
            time_col = actual_time_col if actual_time_col in fe_spec else None
            city_col = actual_city_col if (actual_city_col is not None and actual_city_col in fe_spec) else None

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
                "fixed_effects": " + ".join(fe_spec) if len(fe_spec) > 0 else "无固定效应",
                "did_coefficient": did_res["did_coefficient"],
                "p_value": did_res["p_value"],
                "significant": did_res["significant"],
                "r_squared": did_res["r_squared"],
                "n_obs": did_res["n_obs"],
                "model_type": did_res.get("model_type", "")
            })

        return results

    def run_naive_did(self, policy_name: str, policy_year: int,
                      treat_col: str, outcome_col: str, control_cols: List[str]) -> Dict:
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

        interpretation = self._generate_single_method_interpretation(
            method_name="普通DID",
            policy_name=policy_name,
            policy_year=policy_year,
            outcome_col=outcome_col,
            coef=did_res["did_coefficient"],
            p_value=did_res["p_value"],
            significant=did_res["significant"],
            n_obs=did_res["n_obs"],
            r_squared=did_res["r_squared"]
        )

        return {"did_results": did_res, "llm_interpretation": interpretation}

    def run_psm_did(self, policy_name: str, policy_year: int,
                    treat_col: str, outcome_col: str,
                    control_cols: List[str], match_vars: List[str],
                    caliper: float = 0.05) -> Dict:
        actual_time_col = self.preparer.col_map["time"]
        actual_entity_col = self.preparer.col_map["entity"]

        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)
        df_matched = self.analyzer.psm_matching(df, treat_col, match_vars, caliper=caliper)

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

        interpretation = self._generate_single_method_interpretation(
            method_name="PSM-DID",
            policy_name=policy_name,
            policy_year=policy_year,
            outcome_col=outcome_col,
            coef=did_res["did_coefficient"],
            p_value=did_res["p_value"],
            significant=did_res["significant"],
            n_obs=did_res["n_obs"],
            r_squared=did_res["r_squared"]
        )

        return {"did_results": did_res, "llm_interpretation": interpretation}

    def run_dml(self, policy_name: str, policy_year: int,
                outcome_col: str, treat_col: str, feature_cols: List[str]) -> Dict:
        Y, T, X = self.preparer.prepare_dml_data(outcome_col, treat_col, feature_cols)
        self.dml_analyzer.fit(Y, T, X)
        dml_res = self.dml_analyzer.get_ate_results()

        interpretation = self._generate_single_method_interpretation(
            method_name="双重机器学习（DML）",
            policy_name=policy_name,
            policy_year=policy_year,
            outcome_col=outcome_col,
            coef=dml_res["ate_coefficient"],
            p_value=dml_res["p_value"],
            significant=dml_res["significant"],
            n_obs=len(Y),
            r_squared=None,
            ci_lower=dml_res["ci_lower"],
            ci_upper=dml_res["ci_upper"]
        )

        result = {
            "method": "DML",
            "dml_results": dml_res,
            "sample_size": len(Y),
            "n_features": X.shape[1],
            "llm_interpretation": interpretation
        }
        return result

    def run_event_study(self, policy_name: str, policy_year: int,
                        treat_col: str, outcome_col: str,
                        control_cols: List[str],
                        window_pre: int = 4, window_post: int = 4) -> Dict:
        actual_time_col = self.preparer.col_map["time"]
        actual_entity_col = self.preparer.col_map["entity"]

        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)
        df = self.preparer.prepare_event_study_data(policy_year, treat_col, window_pre, window_post)

        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df.columns else outcome_col
        ctrl_ln = [f"ln_{c}" if f"ln_{c}" in df.columns else c for c in control_cols]

        res = self.analyzer.event_study_regression(
            df=df,
            outcome=outcome_ln,
            controls=ctrl_ln,
            entity_col=actual_entity_col,
            time_col=actual_time_col,
            window_pre=window_pre,
            window_post=window_post
        )

        interpretation = ""
        if self.llm:
            prompt = f"""
请基于事件研究法结果撰写一段政策解读，要求：
1. 说明政策实施前是否存在明显预趋势；
2. 说明政策实施后效应是增强、减弱还是不明显；
3. 语言严谨，不能夸大结论；
4. 输出200-300字。

政策名称：{policy_name}
政策实施年份：{policy_year}
结果变量：{outcome_col}
事件研究结果：
{res['event_results'].to_string(index=False)}
"""
            interpretation = self.llm.generate("政策评估专家", prompt)

        return {
            "event_results": res["event_results"],
            "model_summary": res["model_summary"],
            "llm_interpretation": interpretation
        }

    def run_full_comparison(self, policy_name: str, policy_year: int,
                            treat_col: str, outcome_col: str,
                            control_cols: List[str], match_vars: List[str],
                            dml_features: List[str], caliper: float = 0.05) -> Dict:
        naive_result = self.run_naive_did(policy_name, policy_year, treat_col, outcome_col, control_cols)
        psm_result = self.run_psm_did(policy_name, policy_year, treat_col, outcome_col, control_cols, match_vars, caliper)
        dml_result = self.run_dml(policy_name, policy_year, outcome_col, treat_col, dml_features)

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
        return {"naive_did": naive_result, "psm_did": psm_result, "dml": dml_result, "comparison": comparison}

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
            conclusion += " DML结果显著且控制了高维协变量，结论相对更稳健。"
        elif (naive_sig or psm_sig) and not dml_sig:
            conclusion += " 传统DID显著但DML不显著，表明政策效应可能受到模型设定、遗漏变量或非线性因素影响。"
        elif dml_sig and not (naive_sig or psm_sig):
            conclusion += " DML捕捉到了传统方法未能识别的因果效应，体现了机器学习在复杂关系刻画方面的优势。"
        else:
            conclusion += " 三种方法均不显著，政策效果可能有限，或现有数据与模型对效应识别的支持不足。"

        return conclusion

    def generate_comparison_interpretation(self, comparison: Dict, policy_name: str,
                                           policy_year: int, outcome_col: str) -> str:
        if not self.llm:
            return ""

        if self.llm_style == "论文摘要风格":
            summary_prompt = f"""
请模仿中文经管类学术论文摘要的风格，基于以下政策评估结果，生成一段“综合摘要式解读”。

写作要求：
1. 输出为一个完整自然段，不要分点，不要列标题。
2. 风格要接近学术论文摘要，而不是政府工作报告。
3. 结构上应包含：研究背景、数据与识别思路、核心发现、方法比较、可能机制、异质性或稳健性、研究结论与政策含义。
4. 必须体现普通DID、PSM-DID、DML三种方法的结果对比，并说明它们的一致性与差异。
5. 若三种方法结论不一致，要使用审慎语言，例如“表明政策效应可能受到样本选择、模型设定或非线性因素影响”。
6. 不得虚构不存在的事件研究、机制检验、异质性分析或溢出效应结论；如当前未实际估计，则用“后续可进一步从……展开检验”等规范表述。
7. 输出500-700字，全部使用中文。

已知信息：
- 研究背景：{self.research_background if self.research_background else "面对高质量发展和产业转型升级要求，相关政策已成为推动企业能力提升与绩效改善的重要制度安排。"}
- 政策名称：{policy_name}
- 政策实施年份：{policy_year}
- 结果变量：{outcome_col}
- 可能机制参考：{self.mechanism_hint if self.mechanism_hint else "技术升级、资源配置优化、知识重构、组织协同提升"}
- 异质性参考：{self.heterogeneity_hint if self.heterogeneity_hint else "政策执行能力、知识产权保护、供需关系稳定性、地区制度环境"}

模型结果：
- 普通DID：系数={comparison['comparison']['naive_did']['coefficient']}，p={comparison['comparison']['naive_did']['p_value']}，{'显著' if comparison['comparison']['naive_did']['significant'] else '不显著'}
- PSM-DID：系数={comparison['comparison']['psm_did']['coefficient']}，p={comparison['comparison']['psm_did']['p_value']}，{'显著' if comparison['comparison']['psm_did']['significant'] else '不显著'}
- DML：ATE={comparison['comparison']['dml']['coefficient']}，p={comparison['comparison']['dml']['p_value']}，{'显著' if comparison['comparison']['dml']['significant'] else '不显著'}，95%CI=[{comparison['comparison']['dml']['ci_lower']}, {comparison['comparison']['dml']['ci_upper']}]

请直接输出摘要式解读正文。
"""
        else:
            summary_prompt = f"""
你是专业的政策评估研究员。请基于以下三种方法对【{policy_name}】的评估结果，撰写一份综合政策解读报告，要求严格按照以下结构输出：

一、方法对比
- 对比普通DID、PSM-DID、DML三种方法的估计结果，说明它们的一致性和差异。
- 解释为什么DML的估计值可能更可靠（如控制高维协变量、非线性关系等）。

二、原因分析
- 分析可能产生差异的原因（如选择性偏差、遗漏变量、非线性关系等）。
- 结合政策背景，讨论哪些因素可能影响政策效果的估计。

三、政策建议
请分别从政府和企业两个角度提出具体建议：
（1）政府层面：为了政策应该怎么样（例如：如何优化政策设计、加强监管、扩大覆盖面、配套措施等）。
（2）企业层面：企业根据政策应该怎么做（例如：如何利用政策红利、提升自身能力、参与政策试点等）。

评估结果：
- 普通DID: 系数={comparison['comparison']['naive_did']['coefficient']}, p={comparison['comparison']['naive_did']['p_value']}, {'显著' if comparison['comparison']['naive_did']['significant'] else '不显著'}
- PSM-DID: 系数={comparison['comparison']['psm_did']['coefficient']}, p={comparison['comparison']['psm_did']['p_value']}, {'显著' if comparison['comparison']['psm_did']['significant'] else '不显著'}
- DML: ATE={comparison['comparison']['dml']['coefficient']}, p={comparison['comparison']['dml']['p_value']}, {'显著' if comparison['comparison']['dml']['significant'] else '不显著'}, 95%CI=[{comparison['comparison']['dml']['ci_lower']}, {comparison['comparison']['dml']['ci_upper']}]

请输出500-800字，严格按上述三部分结构。
"""

        return self.llm.generate("政策评估专家", summary_prompt)


# ========== 画图函数 ==========
def plot_parallel_trend(event_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(event_df["event_time"], event_df["coef"], marker="o")
    ax.fill_between(
        event_df["event_time"],
        event_df["ci_lower"],
        event_df["ci_upper"],
        alpha=0.2
    )
    ax.axhline(0, linestyle="--")
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("相对政策时点")
    ax.set_ylabel("估计系数")
    ax.set_title("事件研究法动态效应图")
    st.pyplot(fig)


# ========== Streamlit UI ==========
with st.sidebar:
    st.header("📁 数据上传")
    uploaded_file = st.file_uploader("上传 CSV 或 DTA 文件", type=["csv", "dta"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".dta"):
                df = pd.read_stata(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
            st.session_state.df = df
            st.success(f"数据加载成功：{df.shape[0]}行，{df.shape[1]}列")
        except Exception as e:
            st.error(f"数据加载失败: {e}")

    if st.session_state.df is not None:
        st.header("⚙️ 模型参数")
        cols = st.session_state.df.columns.tolist()

        policy_name = st.text_input("政策名称", value="智能制造试点示范专项行动")
        policy_year = st.number_input("政策实施年份", value=2015, step=1)

        treat_default_index = cols.index("DID") if "DID" in cols else 0
        outcome_default_index = cols.index("RES") if "RES" in cols else 0

        treat_col = st.selectbox("处理组列 (DID)", cols, index=treat_default_index)
        outcome_col = st.selectbox("结果变量", cols, index=outcome_default_index)

        default_ctrl = [c for c in [
            "CRE", "IA", "UD", "AGE", "AGE2", "SIZE", "PROFIT",
            "TOP5", "BOARD", "RD", "FAG", "LE", "OPEN", "GOV"
        ] if c in cols]

        control_cols = st.multiselect("控制变量", cols, default=default_ctrl)
        match_vars = st.multiselect("PSM 匹配变量", cols, default=control_cols)
        dml_features = st.multiselect("DML 特征变量", cols, default=control_cols)

        caliper = st.slider("PSM caliper", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

        st.header("📈 事件研究设置")
        window_pre = st.slider("政策前窗口期", min_value=2, max_value=8, value=4, step=1)
        window_post = st.slider("政策后窗口期", min_value=2, max_value=8, value=4, step=1)

        st.header("🧠 LLM 解读设置")
        llm_style = st.selectbox(
            "LLM解读风格",
            ["论文摘要风格", "政策报告风格"],
            index=0
        )

        research_background = st.text_area(
            "研究背景（用于摘要生成）",
            value=(
                "面对新一轮科技革命和产业变革，相关政策已成为推动企业转型升级、"
                "提升产业链供应链韧性和促进高质量发展的重要制度安排。"
            ),
            height=120
        )

        mechanism_hint = st.text_input(
            "可能机制提示（可选）",
            value="技术升级、资源配置优化、知识重构、组织协同提升"
        )

        heterogeneity_hint = st.text_input(
            "异质性提示（可选）",
            value="政策执行能力、知识产权保护、供需关系稳定性、地区制度环境"
        )

        st.header("🔑 LLM 配置")
        api_key = st.text_input(
            "通义千问 API Key",
            type="password",
            help="输入您的 DashScope API Key，留空则跳过 LLM 解读"
        )

        run_button = st.button("🚀 运行评估", type="primary")


# 主区域
if st.session_state.df is not None:
    st.subheader("🗂 数据预览")
    st.dataframe(st.session_state.df.head(20), use_container_width=True)

if st.session_state.df is not None and run_button:
    with st.spinner("正在分析数据..."):
        llm = None

        if api_key:
            try:
                llm = QwenEngine(api_key=api_key)
                test_resp = llm.generate("测试", "请只回复：OK")
                if "API 调用失败" in test_resp or "系统错误" in test_resp:
                    st.warning(f"LLM 测试失败: {test_resp}，将跳过 LLM 解读。")
                    llm = None
                else:
                    st.success("LLM 连接成功")
            except Exception as e:
                st.warning(f"LLM 初始化失败: {e}，将跳过 LLM 解读。")
                llm = None
        else:
            st.info("未提供 API Key，将跳过 LLM 解读。")

        try:
            pipeline = PolicyEvaluationPipeline(
                st.session_state.df,
                llm_engine=llm,
                llm_style=llm_style,
                research_background=research_background,
                mechanism_hint=mechanism_hint,
                heterogeneity_hint=heterogeneity_hint
            )

            # 1. 多规格 DID
            st.subheader("📈 普通DID多规格回归")
            actual_entity_col = pipeline.preparer.col_map["entity"]
            actual_time_col = pipeline.preparer.col_map["time"]
            actual_city_col = pipeline.preparer.col_map.get("city", None)

            fe_specs = [
                [],
                [actual_entity_col],
                [actual_entity_col, actual_time_col]
            ]
            if actual_city_col is not None:
                fe_specs.append([actual_entity_col, actual_time_col, actual_city_col])

            multi_results = pipeline.run_naive_did_with_multiple_specs(
                policy_year, treat_col, outcome_col, control_cols, fe_specs
            )
            df_multi = pd.DataFrame(multi_results)
            st.dataframe(df_multi, use_container_width=True)

            # 2. 普通DID
            st.subheader("📊 普通DID（企业 + 年份固定效应）")
            naive_res = pipeline.run_naive_did(policy_name, policy_year, treat_col, outcome_col, control_cols)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("DID系数", naive_res["did_results"]["did_coefficient"])
            col2.metric("P值", naive_res["did_results"]["p_value"])
            col3.metric("显著", "✅ 是" if naive_res["did_results"]["significant"] else "❌ 否")
            col4.metric("样本量", naive_res["did_results"]["n_obs"])

            if llm and naive_res["llm_interpretation"]:
                with st.expander("📝 普通DID的LLM解读"):
                    st.write(naive_res["llm_interpretation"])

            # 3. PSM-DID
            st.subheader("🔗 PSM-DID")
            psm_res = pipeline.run_psm_did(
                policy_name, policy_year, treat_col, outcome_col, control_cols, match_vars, caliper=caliper
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("DID系数", psm_res["did_results"]["did_coefficient"])
            col2.metric("P值", psm_res["did_results"]["p_value"])
            col3.metric("显著", "✅ 是" if psm_res["did_results"]["significant"] else "❌ 否")
            col4.metric("样本量", psm_res["did_results"]["n_obs"])

            if llm and psm_res["llm_interpretation"]:
                with st.expander("📝 PSM-DID的LLM解读"):
                    st.write(psm_res["llm_interpretation"])

            # 4. DML
            st.subheader("🤖 双重机器学习（DML）")
            dml_res = pipeline.run_dml(policy_name, policy_year, outcome_col, treat_col, dml_features)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ATE系数", dml_res["dml_results"]["ate_coefficient"])
            col2.metric("标准误", dml_res["dml_results"]["std_error"])
            col3.metric("P值", dml_res["dml_results"]["p_value"])
            col4.metric("显著", "✅ 是" if dml_res["dml_results"]["significant"] else "❌ 否")
            st.caption(
                f"95%置信区间: [{dml_res['dml_results']['ci_lower']}, {dml_res['dml_results']['ci_upper']}]"
            )

            if llm and dml_res["llm_interpretation"]:
                with st.expander("📝 DML的LLM解读"):
                    st.write(dml_res["llm_interpretation"])

            # 5. 事件研究法
            st.subheader("📉 事件研究法（Event Study）")
            event_res = pipeline.run_event_study(
                policy_name, policy_year, treat_col, outcome_col, control_cols,
                window_pre=window_pre, window_post=window_post
            )
            st.dataframe(event_res["event_results"], use_container_width=True)
            plot_parallel_trend(event_res["event_results"])

            if llm and event_res["llm_interpretation"]:
                with st.expander("📝 事件研究法的LLM解读"):
                    st.write(event_res["llm_interpretation"])

            # 6. 对比表格
            st.subheader("📋 三种方法对比")
            comparison = pipeline.run_full_comparison(
                policy_name, policy_year, treat_col, outcome_col,
                control_cols, match_vars, dml_features, caliper=caliper
            )

            comparison_df = pd.DataFrame({
                "方法": ["普通DID", "PSM-DID", "DML"],
                "系数": [
                    comparison["comparison"]["naive_did"]["coefficient"],
                    comparison["comparison"]["psm_did"]["coefficient"],
                    comparison["comparison"]["dml"]["coefficient"]
                ],
                "P值": [
                    comparison["comparison"]["naive_did"]["p_value"],
                    comparison["comparison"]["psm_did"]["p_value"],
                    comparison["comparison"]["dml"]["p_value"]
                ],
                "显著": [
                    "✅" if comparison["comparison"]["naive_did"]["significant"] else "❌",
                    "✅" if comparison["comparison"]["psm_did"]["significant"] else "❌",
                    "✅" if comparison["comparison"]["dml"]["significant"] else "❌"
                ]
            })
            st.dataframe(comparison_df, use_container_width=True)
            st.info(comparison["comparison"]["conclusion"])

            # 7. 综合解读
            if llm:
                st.subheader("🧠 综合政策解读")
                with st.spinner("正在生成综合解读..."):
                    final_report = pipeline.generate_comparison_interpretation(
                        comparison=comparison,
                        policy_name=policy_name,
                        policy_year=policy_year,
                        outcome_col=outcome_col
                    )
                    st.markdown(final_report)
            else:
                st.info("未启用 LLM，无法生成综合解读。")

            # 8. 导出结果
            st.subheader("📥 结果导出")
            export_dict = {
                "multi_spec_did": df_multi.to_dict(orient="records"),
                "naive_did": naive_res["did_results"],
                "psm_did": psm_res["did_results"],
                "dml": dml_res["dml_results"],
                "event_study": event_res["event_results"].to_dict(orient="records"),
                "comparison": comparison["comparison"]
            }
            export_json = pd.Series(export_dict).to_json(force_ascii=False, indent=2)
            st.download_button(
                label="下载结果（JSON）",
                data=export_json,
                file_name="policy_evaluation_results.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"运行评估失败：{e}")

elif st.session_state.df is None:
    st.info("👈 请从左侧上传数据文件并配置参数，然后点击“运行评估”")
