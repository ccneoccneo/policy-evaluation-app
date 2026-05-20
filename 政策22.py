# app.py（最终版，综合解读严格遵循学术摘要格式）
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Dict, List, Tuple
import dashscope
from http import HTTPStatus
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy import stats

st.set_page_config(page_title="政策效果评估系统", layout="wide")
st.title("📊 政策效果自动评估系统")
st.markdown("支持普通DID、PSM-DID、双重机器学习（DML），集成通义千问LLM解读")

# ---------- 初始化 ----------
if "results" not in st.session_state:
    st.session_state.results = {}
if "df" not in st.session_state:
    st.session_state.df = None


# ========== 通义千问引擎 ==========
class QwenEngine:
    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        if not api_key:
            raise ValueError("API Key 不能为空")
        self.api_key = api_key
        self.model = model
        dashscope.api_key = api_key

    def generate(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 800, temperature: float = 0.4) -> str:
        full_prompt = f"系统指令：{system_prompt}\n\n用户问题：{user_prompt}"
        return self._safe_api_call(full_prompt, max_tokens, temperature)

    def _safe_api_call(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的政策评估研究员，擅长将计量经济学结果转化为政府研究报告风格的语言。请用中文回答。"},
                {"role": "user", "content": prompt}
            ]
            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                result_format='message'
            )
            if response.status_code != HTTPStatus.OK:
                return f"API 调用失败: {response.code} - {response.message}"
            if (hasattr(response, 'output') and response.output and
                hasattr(response.output, 'choices') and response.output.choices and
                len(response.output.choices) > 0):
                choice = response.output.choices[0]
                if hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    if content:
                        return content
            if hasattr(response, 'output') and hasattr(response.output, 'text'):
                return response.output.text
            return "API 返回数据格式异常，无法提取内容。"
        except Exception as e:
            return f"系统错误: {str(e)}"


# ========== 数据准备模块 ==========
class PolicyDataPreparer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
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
        else:
            self.col_map["city"] = None

    def prepare_did_data(self, policy_year: int, treat_col: str,
                         outcome_col: str, control_cols: List[str]) -> pd.DataFrame:
        df = self.df.copy()
        actual_time_col = self.col_map["time"]
        df["post"] = (df[actual_time_col] >= policy_year).astype(int)
        df["did_interaction"] = df[treat_col] * df["post"]
        for col in [outcome_col] + control_cols:
            if col in df.columns and df[col].min() > 0:
                df[f"ln_{col}"] = np.log(df[col])
        return df

    def prepare_dml_data(self, outcome_col: str, treat_col: str,
                         feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = self.df.copy()
        actual_entity_col = self.col_map["entity"]
        for col in [outcome_col, treat_col] + feature_cols:
            if col in df.columns:
                df[f"{col}_demeaned"] = df.groupby(actual_entity_col)[col].transform(lambda x: x - x.mean())
        Y = df[f"{outcome_col}_demeaned"].values
        T = df[f"{treat_col}_demeaned"].values
        X = df[[f"{c}_demeaned" for c in feature_cols if c in df.columns]].values
        mask = ~(np.isnan(Y) | np.isnan(T) | np.isnan(X).any(axis=1))
        Y, T, X = Y[mask], T[mask], X[mask]
        X = StandardScaler().fit_transform(X)
        return Y, T, X


# ========== 计量分析模块 ==========
class CausalInferenceAnalyzer:
    @staticmethod
    def psm_matching(df: pd.DataFrame, treat_col: str, match_vars: List[str], caliper=0.05) -> pd.DataFrame:
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
        return matched_df

    @staticmethod
    def did_regression(df: pd.DataFrame, outcome: str,
                       did_col="did_interaction", treat_col="treat",
                       post_col="post", controls: List[str] = None,
                       entity_col=None, time_col=None, city_col=None) -> Dict:
        try:
            from linearmodels.panel import PanelOLS
            fe_terms = []
            if entity_col and entity_col in df.columns:
                fe_terms.append("EntityEffects")
            if time_col and time_col in df.columns:
                fe_terms.append("TimeEffects")
            if city_col and city_col in df.columns:
                city_dummies = pd.get_dummies(df[city_col], prefix='city', drop_first=True)
                df = df.copy()
                df = pd.concat([df, city_dummies], axis=1)
                city_controls = list(city_dummies.columns)
                if controls:
                    controls = controls + city_controls
                else:
                    controls = city_controls
            df_panel = df.set_index([entity_col, time_col])
            ctrl_str = " + ".join(controls) if controls else ""
            fe_str = " + ".join(fe_terms)
            if fe_str:
                formula = f"{outcome} ~ {did_col} + {ctrl_str} + {fe_str}"
            else:
                formula = f"{outcome} ~ {did_col} + {ctrl_str}"
            model = PanelOLS.from_formula(formula, data=df_panel, drop_absorbed=True)
            res = model.fit(cov_type="clustered", cluster_entity=True)
        except Exception:
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


# ========== 双重机器学习模块 ==========
class DoubleMachineLearningAnalyzer:
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self._fitted = False
        self._ate = None
        self._ate_std = None
        self._ate_pvalue = None
        self._ate_ci = None

    def _get_default_models(self):
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42), \
               RidgeCV(alphas=[0.1, 1.0, 10.0])

    def _t_stat_pvalue(self, t_value, df=100):
        return 2 * (1 - stats.t.cdf(abs(t_value), df=df))

    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray):
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
        return self

    def get_ate_results(self) -> Dict:
        if not self._fitted:
            raise ValueError("模型未拟合")
        return {
            "ate_coefficient": round(float(self._ate), 4),
            "std_error": round(float(self._ate_std), 4),
            "p_value": round(float(self._ate_pvalue), 4),
            "significant": self._ate_pvalue < 0.05,
            "ci_lower": round(float(self._ate_ci[0]), 4),
            "ci_upper": round(float(self._ate_ci[1]), 4),
            "method": "Double Machine Learning (DML)",
        }


# ========== 评估流水线 ==========
class PolicyEvaluationPipeline:
    def __init__(self, df: pd.DataFrame, llm_engine=None):
        self.preparer = PolicyDataPreparer(df)
        self.analyzer = CausalInferenceAnalyzer()
        self.dml_analyzer = DoubleMachineLearningAnalyzer()
        self.llm = llm_engine

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
            entity_col = actual_entity_col if 'stkcd' in fe_spec or actual_entity_col in fe_spec else None
            time_col = actual_time_col if 'year' in fe_spec else None
            city_col = actual_city_col if 'city' in fe_spec else None
            did_res = self.analyzer.did_regression(
                df, outcome_ln, did_col="did_interaction", treat_col=treat_col, post_col="post",
                controls=ctrl_ln, entity_col=entity_col, time_col=time_col, city_col=city_col
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

    def run_naive_did(self, policy_name: str, policy_year: int,
                      treat_col: str, outcome_col: str, control_cols: List[str]) -> Dict:
        actual_time_col = self.preparer.col_map["time"]
        actual_entity_col = self.preparer.col_map["entity"]
        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)
        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df.columns else outcome_col
        ctrl_ln = [f"ln_{c}" if f"ln_{c}" in df.columns else c for c in control_cols]
        did_res = self.analyzer.did_regression(
            df, outcome_ln, did_col="did_interaction", treat_col=treat_col, post_col="post",
            controls=ctrl_ln, entity_col=actual_entity_col, time_col=actual_time_col
        )
        if self.llm:
            prompt = f"""
你是专业的政策评估研究员。请基于普通DID方法的估计结果，撰写一份**具体、机制明确**的政策解读。要求：

1. 明确说明政策名称是【{policy_name}】，结果变量是【{outcome_col}】。
2. 总结政策效果的方向、大小和统计显著性，并量化解释。
3. 分析政策发挥作用的具体机制，包括降低成本、提升技术能力、优化流程、促进知识重构等。
4. 提出3条具体、可操作的政策建议，必须分别从政府和企业两个角度给出。
5. 避免泛泛而谈。

估计结果：DID系数={did_res['did_coefficient']}，P值={did_res['p_value']}（{'显著' if did_res['significant'] else '不显著'}），样本量={did_res['n_obs']}，R²={did_res['r_squared']}。

输出400-500字。
"""
            interpretation = self.llm.generate("政策评估专家", prompt)
        else:
            interpretation = ""
        return {"did_results": did_res, "llm_interpretation": interpretation}

    def run_psm_did(self, policy_name: str, policy_year: int,
                    treat_col: str, outcome_col: str,
                    control_cols: List[str], match_vars: List[str]) -> Dict:
        actual_time_col = self.preparer.col_map["time"]
        actual_entity_col = self.preparer.col_map["entity"]
        df = self.preparer.prepare_did_data(policy_year, treat_col, outcome_col, control_cols)
        df_matched = self.analyzer.psm_matching(df, treat_col, match_vars)
        outcome_ln = f"ln_{outcome_col}" if f"ln_{outcome_col}" in df_matched.columns else outcome_col
        ctrl_ln = [f"ln_{c}" if f"ln_{c}" in df_matched.columns else c for c in control_cols]
        did_res = self.analyzer.did_regression(
            df_matched, outcome_ln, did_col="did_interaction", treat_col=treat_col, post_col="post",
            controls=ctrl_ln, entity_col=actual_entity_col, time_col=actual_time_col
        )
        if self.llm:
            prompt = f"""
你是专业的政策评估研究员。请基于PSM-DID方法的估计结果，撰写一份**具体、机制明确**的政策解读。要求：

1. 明确说明政策名称是【{policy_name}】，结果变量是【{outcome_col}】。
2. 总结政策效果的方向、大小和统计显著性，并量化解释。
3. 分析政策发挥作用的具体机制，包括降低成本、提升技术能力、优化流程、促进知识重构等。
4. 提出3条具体、可操作的政策建议，必须分别从政府和企业两个角度给出。
5. 避免泛泛而谈。

估计结果：DID系数={did_res['did_coefficient']}，P值={did_res['p_value']}（{'显著' if did_res['significant'] else '不显著'}），样本量={did_res['n_obs']}，R²={did_res['r_squared']}。

输出400-500字。
"""
            interpretation = self.llm.generate("政策评估专家", prompt)
        else:
            interpretation = ""
        return {"did_results": did_res, "llm_interpretation": interpretation}

    def run_dml(self, policy_name: str, outcome_col: str, treat_col: str, feature_cols: List[str]) -> Dict:
        Y, T, X = self.preparer.prepare_dml_data(outcome_col, treat_col, feature_cols)
        self.dml_analyzer.fit(Y, T, X)
        dml_res = self.dml_analyzer.get_ate_results()
        if self.llm:
            prompt = f"""
请基于DML估计结果撰写政策解读，要求具体、机制明确：
- ATE系数：{dml_res.get('ate_coefficient', 'N/A')}
- 标准误：{dml_res.get('std_error', 'N/A')}
- P值：{dml_res.get('p_value', 'N/A')}（{'显著' if dml_res.get('significant', False) else '不显著'}）
- 95%置信区间：[{dml_res.get('ci_lower', 'N/A')}, {dml_res.get('ci_upper', 'N/A')}]

要求：
1. 解释ATE的含义。
2. 说明DML相比传统方法的优势。
3. 提出3条具体优化建议，分别从政府和企业角度。
输出400-500字。
"""
            interpretation = self.llm.generate("政策评估专家", prompt)
        else:
            interpretation = ""
        result = {"method": "DML", "dml_results": dml_res, "sample_size": len(Y), "n_features": X.shape[1],
                  "llm_interpretation": interpretation}
        return result

    def run_full_comparison(self, policy_name: str, policy_year: int,
                            treat_col: str, outcome_col: str,
                            control_cols: List[str], match_vars: List[str],
                            dml_features: List[str]) -> Dict:
        naive_result = self.run_naive_did(policy_name, policy_year, treat_col, outcome_col, control_cols)
        psm_result = self.run_psm_did(policy_name, policy_year, treat_col, outcome_col, control_cols, match_vars)
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
            conclusion += " DML结果显著且控制了高维协变量，结论更可信。"
        elif (naive_sig or psm_sig) and not dml_sig:
            conclusion += " 传统DID显著但DML不显著，可能存在遗漏变量或非线性关系，建议以DML为准。"
        elif dml_sig and not (naive_sig or psm_sig):
            conclusion += " DML捕捉到了传统方法未能发现的因果效应，体现了机器学习优势。"
        else:
            conclusion += " 三种方法均不显著，政策效果可能有限或数据不足。"
        return conclusion


# ========== Streamlit UI ==========
with st.sidebar:
    st.header("📁 数据上传")
    uploaded_file = st.file_uploader("上传 CSV 或 DTA 文件", type=["csv", "dta"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.dta'):
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
        policy_year = st.number_input("政策实施年份", value=2015, step=1)
        treat_col = st.selectbox("处理组列 (DID)", cols, index=cols.index("DID") if "DID" in cols else 0)
        outcome_col = st.selectbox("结果变量", cols, index=cols.index("RES") if "RES" in cols else 0)
        default_ctrl = [c for c in ["CRE","IA","UD","AGE","AGE2","SIZE","PROFIT","TOP5","BOARD","RD","FAG","LE","OPEN","GOV"] if c in cols]
        control_cols = st.multiselect("控制变量", cols, default=default_ctrl)
        match_vars = st.multiselect("PSM 匹配变量", cols, default=control_cols)
        dml_features = st.multiselect("DML 特征变量", cols, default=control_cols)

        st.header("🔑 LLM 配置")
        api_key = st.text_input("通义千问 API Key", type="password", help="输入您的 DashScope API Key，留空则跳过 LLM 解读")
        run_button = st.button("🚀 运行评估", type="primary")

POLICY_NAME = "智能制造政策"

# 主区域
if st.session_state.df is not None and run_button:
    with st.spinner("正在分析数据..."):
        llm = None
        if api_key:
            try:
                llm = QwenEngine(api_key=api_key)
                test_resp = llm.generate("测试", "请回复 OK")
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

        pipeline = PolicyEvaluationPipeline(st.session_state.df, llm_engine=llm)

        # 多规格 DID
        st.subheader("📈 普通DID多规格回归")
        fe_specs = [[], ['stkcd'], ['stkcd', 'year'], ['stkcd', 'year', 'city']]
        multi_results = pipeline.run_naive_did_with_multiple_specs(
            policy_year, treat_col, outcome_col, control_cols, fe_specs
        )
        st.dataframe(pd.DataFrame(multi_results), use_container_width=True)

        # 普通DID
        st.subheader("📊 普通DID (企业+年份固定效应)")
        naive_res = pipeline.run_naive_did(POLICY_NAME, policy_year, treat_col, outcome_col, control_cols)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DID系数", naive_res["did_results"]["did_coefficient"])
        c2.metric("P值", naive_res["did_results"]["p_value"])
        c3.metric("显著", "✅" if naive_res["did_results"]["significant"] else "❌")
        c4.metric("样本量", naive_res["did_results"]["n_obs"])
        if llm and naive_res["llm_interpretation"]:
            with st.expander("📝 普通DID的LLM解读"):
                st.write(naive_res["llm_interpretation"])

        # PSM-DID
        st.subheader("🔗 PSM-DID")
        psm_res = pipeline.run_psm_did(POLICY_NAME, policy_year, treat_col, outcome_col, control_cols, match_vars)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DID系数", psm_res["did_results"]["did_coefficient"])
        c2.metric("P值", psm_res["did_results"]["p_value"])
        c3.metric("显著", "✅" if psm_res["did_results"]["significant"] else "❌")
        c4.metric("样本量", psm_res["did_results"]["n_obs"])
        if llm and psm_res["llm_interpretation"]:
            with st.expander("📝 PSM-DID的LLM解读"):
                st.write(psm_res["llm_interpretation"])

        # DML
        st.subheader("🤖 双重机器学习 (DML)")
        dml_res = pipeline.run_dml(POLICY_NAME, outcome_col, treat_col, dml_features)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATE系数", dml_res["dml_results"]["ate_coefficient"])
        c2.metric("标准误", dml_res["dml_results"]["std_error"])
        c3.metric("P值", dml_res["dml_results"]["p_value"])
        c4.metric("显著", "✅" if dml_res["dml_results"]["significant"] else "❌")
        st.caption(f"95%置信区间: [{dml_res['dml_results']['ci_lower']}, {dml_res['dml_results']['ci_upper']}]")
        if llm and dml_res["llm_interpretation"]:
            with st.expander("📝 DML的LLM解读"):
                st.write(dml_res["llm_interpretation"])

        # 对比表格
        st.subheader("📋 三种方法对比")
        comparison = pipeline.run_full_comparison(
            POLICY_NAME, policy_year, treat_col, outcome_col,
            control_cols, match_vars, dml_features
        )
        comp_df = pd.DataFrame({
            "方法": ["普通DID", "PSM-DID", "DML"],
            "系数": [comparison["comparison"]["naive_did"]["coefficient"],
                     comparison["comparison"]["psm_did"]["coefficient"],
                     comparison["comparison"]["dml"]["coefficient"]],
            "P值": [comparison["comparison"]["naive_did"]["p_value"],
                    comparison["comparison"]["psm_did"]["p_value"],
                    comparison["comparison"]["dml"]["p_value"]],
            "显著": ["✅" if comparison["comparison"]["naive_did"]["significant"] else "❌",
                     "✅" if comparison["comparison"]["psm_did"]["significant"] else "❌",
                     "✅" if comparison["comparison"]["dml"]["significant"] else "❌"]
        })
        st.dataframe(comp_df, use_container_width=True)

        # 综合解读（学术摘要风格，降低 temperature 增强格式遵循）
        if llm:
            st.subheader("🧠 综合政策解读报告（学术摘要风格）")
            with st.spinner("正在生成综合报告..."):
                summary_prompt = f"""
你是专业的政策评估研究员。请基于以下三种方法对【{POLICY_NAME}】的评估结果，撰写一份**学术论文摘要风格**的综合政策解读报告。

【严格要求】：
1. 必须使用以下七个带方括号的标题，且顺序不可改变：
   【背景与问题】、【研究方法】、【主要发现】、【作用机制】、【进一步分析】、【结论与建议】
2. 禁止使用“一、二、三”或“1. 2. 3.”等编号。
3. 每个标题后直接跟内容，内容需简洁、正式、逻辑清晰。
4. 在【结论与建议】中，必须分别从政府和企业两个角度提出具体建议。

【评估结果】：
- 普通DID: 系数={comparison['comparison']['naive_did']['coefficient']}, p={comparison['comparison']['naive_did']['p_value']}, {'显著' if comparison['comparison']['naive_did']['significant'] else '不显著'}, 样本量={comparison['comparison']['naive_did']['n_obs']}
- PSM-DID: 系数={comparison['comparison']['psm_did']['coefficient']}, p={comparison['comparison']['psm_did']['p_value']}, {'显著' if comparison['comparison']['psm_did']['significant'] else '不显著'}, 样本量={comparison['comparison']['psm_did']['n_obs']}
- DML: ATE={comparison['comparison']['dml']['coefficient']}, p={comparison['comparison']['dml']['p_value']}, {'显著' if comparison['comparison']['dml']['significant'] else '不显著'}, 95%CI=[{comparison['comparison']['dml']['ci_lower']}, {comparison['comparison']['dml']['ci_upper']}]

请输出500-800字，严格按照上述七个带方括号的标题格式。开始输出：
"""
                # 降低 temperature 以增强格式遵循
                final_report = llm.generate("政策评估专家", summary_prompt, temperature=0.2)
                st.markdown(final_report)
        else:
            st.info("未启用 LLM，无法生成综合解读。")

elif st.session_state.df is None:
    st.info("👈 请从左侧上传数据文件并配置参数，然后点击「运行评估」")