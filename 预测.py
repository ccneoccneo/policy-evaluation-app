import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
import matplotlib.pyplot as plt
import matplotlib
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
# -----------------------------
# 1. 生成模拟数据
# -----------------------------
packaging_list = ['Normal','Vacuum','MAP']
temperature_list = [277.15, 283.15, 298.15]  # 4C, 10C, 25C
time_points = np.arange(0,6)  # 0~5天
indicators = ['ColonyCount','Hardness','Elasticity','Chewiness','Adhesiveness']

np.random.seed(42)
data = []

for pack in packaging_list:
    for T in temperature_list:
        B0_dict = {
            'ColonyCount': np.random.uniform(3,4),
            'Hardness': np.random.uniform(15,17),
            'Elasticity': np.random.uniform(0.8,0.9),
            'Chewiness': np.random.uniform(0.6,0.7),
            'Adhesiveness': np.random.uniform(0.1,0.2)
        }
        k_dict = {
            'ColonyCount': np.random.uniform(0.05,0.1),
            'Hardness': np.random.uniform(-0.2,-0.1),
            'Elasticity': np.random.uniform(-0.02,-0.01),
            'Chewiness': np.random.uniform(-0.015,-0.005),
            'Adhesiveness': np.random.uniform(-0.005,-0.001)
        }
        for t in time_points:
            row = {'Packaging':pack,'Temperature':T,'Time':t}
            for ind in indicators:
                val = B0_dict[ind] + k_dict[ind]*t
                val += np.random.normal(0,0.02)
                row[ind] = val
            data.append(row)

df = pd.DataFrame(data)

# -----------------------------
# 2. 定义动力学和Arrhenius函数
# -----------------------------
def zero_order(t, B0, k):
    return B0 + k*t

def k_arrhenius(T, k0, Ea):
    R = 8.314
    return k0 * np.exp(-Ea / (R*T))

# -----------------------------
# 3. 拟合零级动力学
# -----------------------------
k_values = {}
B0_values = {}

for pack in packaging_list:
    for ind in indicators:
        for T in temperature_list:
            df_sub = df[(df['Packaging']==pack) & (df['Temperature']==T)]
            popt, _ = curve_fit(zero_order, df_sub['Time'], df_sub[ind])
            B0, k = popt
            k_values[(pack,ind,T)] = k
            B0_values[(pack,ind,T)] = B0

# -----------------------------
# 4. Arrhenius拟合
# -----------------------------
Ea_k0_dict = {}

for pack in packaging_list:
    for ind in indicators:
        ks = np.array([k_values[(pack,ind,T)] for T in temperature_list])
        lnk = np.log(np.abs(ks))
        invT = 1/np.array(temperature_list)
        model = LinearRegression()
        model.fit(invT.reshape(-1,1), lnk)
        slope = model.coef_[0]
        intercept = model.intercept_
        R = 8.314
        Ea = -slope * R
        k0 = np.exp(intercept)
        Ea_k0_dict[(pack,ind)] = (Ea,k0)

# -----------------------------
# 5. 预测货架期
# -----------------------------
B_limit = {
    'ColonyCount':5.0,
    'Hardness':12.0,
    'Elasticity':0.75,
    'Chewiness':0.55,
    'Adhesiveness':0.12
}

Q_results = []

for pack in packaging_list:
    for T in temperature_list:
        Q_indicators = []
        for ind in indicators:
            B0 = B0_values[(pack,ind,T)]
            Ea,k0 = Ea_k0_dict[(pack,ind)]
            k_T = k_arrhenius(T, k0, Ea)
            if B_limit[ind] > B0:
                Q = (B_limit[ind]-B0)/k_T
            else:
                Q = (B0-B_limit[ind])/abs(k_T)
            Q_indicators.append(Q)
        Q_total = min(Q_indicators)
        row = {
            'Packaging':pack,
            'Temperature':T,
            'Q_ColonyCount':Q_indicators[0],
            'Q_Hardness':Q_indicators[1],
            'Q_Elasticity':Q_indicators[2],
            'Q_Chewiness':Q_indicators[3],
            'Q_Adhesiveness':Q_indicators[4],
            'Q_Total':Q_total
        }
        Q_results.append(row)

df_Q = pd.DataFrame(Q_results)

# -----------------------------
# 6. 可视化：时间-指标曲线
# -----------------------------
for ind in indicators:
    plt.figure(figsize=(8,5))
    for pack in packaging_list:
        df_plot = df[(df['Packaging']==pack) & (df['Temperature']==277.15)]
        plt.plot(df_plot['Time'], df_plot[ind], marker='o', label=f"{pack}")
    plt.xlabel("Time (days)")
    plt.ylabel(ind)
    plt.title(f"4℃下不同包装 {ind} 随时间变化")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------
# 7. 可视化：货架期对比柱状图（综合指标）
# -----------------------------
for T in temperature_list:
    df_bar = df_Q[df_Q['Temperature']==T]
    plt.figure(figsize=(8,5))
    plt.bar(df_bar['Packaging'], df_bar['Q_Total'], color=['orange','skyblue','green'])
    plt.ylabel("Predicted Shelf Life (days)")
    plt.title(f"{int(T-273.15)}℃下不同包装货架期对比（综合指标）")
    plt.show()