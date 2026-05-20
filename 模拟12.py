"""
“AI+物流系统仿真”完整案例（第三次修正——根本性修复）
修复逻辑：冲击发生后，到达速率立即恢复到正常水平，消除无限积压的根源。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# ==================== 系统动力学仿真（修正版） ====================
def logistics_hub_simulation(total_days=60, disruption_day=20, disruption_severity=0.5,
                             initial_inventory=500, arrival_spike=1.5, recovery_rate=0.15):
    """
    物流枢纽韧性仿真（修正版：冲击后到达立即正常）
    """
    normal_arrival_rate = 100
    normal_processing_rate = 105
    normal_level = initial_inventory * 1.0

    inventory = np.zeros(total_days)
    processed = np.zeros(total_days)
    arrival = np.zeros(total_days)
    inventory[0] = initial_inventory

    for t in range(total_days):
        # ----- 到达量 -----
        if 10 <= t <= 15:                # 旺季高峰
            arrival[t] = normal_arrival_rate * arrival_spike
        else:
            arrival[t] = normal_arrival_rate   # 其余时间正常（包括冲击后）

        # ----- 处理能力 -----
        if t < disruption_day:
            processing_capacity = normal_processing_rate
        else:
            if t == disruption_day:
                capacity_factor = 1 - disruption_severity
            else:
                recovery = 1 - disruption_severity * np.exp(-recovery_rate * (t - disruption_day))
                capacity_factor = recovery
            capacity_factor = max(capacity_factor, 0.15)
            processing_capacity = normal_processing_rate * capacity_factor

        processed[t] = min(processing_capacity, inventory[t] + arrival[t])

        if t < total_days - 1:
            inventory[t+1] = max(0, inventory[t] + arrival[t] - processed[t])

    # ----- 恢复天数 -----
    recovery_days = 999
    for i in range(disruption_day + 1, total_days):
        if inventory[i] <= normal_level * 1.05:
            recovery_days = i - disruption_day
            break

    max_inventory = np.max(inventory[disruption_day:])
    return inventory, processed, arrival, recovery_days, max_inventory

# 测试
inv, proc, arr, rec, max_inv = logistics_hub_simulation(disruption_severity=0.5, arrival_spike=1.5)
print(f"示例仿真: 恢复天数={rec}, 最大库存={max_inv:.0f}")

# ==================== 生成数据集 ====================
def generate_dataset(n_samples=500):
    np.random.seed(42)
    data = []
    for _ in range(n_samples):
        sev = np.random.uniform(0.2, 0.7)
        spike = np.random.uniform(1.2, 1.8)
        init_inv = np.random.randint(300, 600)
        rec_rate = np.random.uniform(0.05, 0.25)
        _, _, _, rec_days, max_inv = logistics_hub_simulation(
            disruption_severity=sev, arrival_spike=spike,
            initial_inventory=init_inv, recovery_rate=rec_rate)
        data.append([sev, spike, init_inv, rec_rate, rec_days, max_inv])

    df = pd.DataFrame(data, columns=['severity', 'arrival_spike', 'initial_inventory',
                                     'recovery_rate', 'recovery_days', 'max_inventory'])
    return df

df = generate_dataset(500)
print(f"恢复天数分布: min={df['recovery_days'].min()}, max={df['recovery_days'].max()}, mean={df['recovery_days'].mean():.1f}")
print(f"999 占比: {np.mean(df['recovery_days']==999)*100:.1f}%")
print(f"0 占比: {np.mean(df['recovery_days']==0)*100:.1f}%")

# 仅保留可恢复样本
df_valid = df[df['recovery_days'] < 999].copy()
print(f"有效样本数: {len(df_valid)}")

# ==================== 机器学习 ====================
X = df_valid[['severity', 'arrival_spike', 'initial_inventory', 'recovery_rate']]
y = df_valid['recovery_days']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"随机森林 MAE: {mae:.2f} 天")

# 特征重要性
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n特征重要性：")
print(importance)

# ==================== 产教融合应用 ====================
print("\n" + "="*60)
print("产教融合场景：与重庆物流集团合作，应急响应预案优化")
print("="*60)

current_params = {
    'severity': 0.55,
    'arrival_spike': 1.5,
    'initial_inventory': 450,
    'recovery_rate': 0.12
}
baseline_rec = model.predict(pd.DataFrame([current_params]))[0]
print(f"【当前预案】预测恢复天数: {baseline_rec:.1f} 天")

plans = [
    ('基础预案', 0.12, '仅自有资源'),
    ('方案A：外部设备租赁', 0.22, '与设备租赁公司签订应急协议'),
    ('方案B：设备+人力双重保障', 0.32, '额外配备应急抢修队伍并租赁设备'),
]

print("\n--- 方案对比 ---")
plan_names, plan_recs = [], []
for name, rate, desc in plans:
    params = current_params.copy()
    params['recovery_rate'] = rate
    pred = model.predict(pd.DataFrame([params]))[0]
    plan_names.append(name)
    plan_recs.append(pred)
    print(f"  {name}: 恢复速率={rate:.2f}, 预测恢复 {pred:.1f} 天")

best_idx = np.argmin(plan_recs)
print(f"\n【推荐】采用 {plan_names[best_idx]}，预计恢复时间：{plan_recs[best_idx]:.1f} 天")

# 可视化
fig, ax = plt.subplots(figsize=(8,5))
colors = ['gray','orange','green']
bars = ax.bar(plan_names, plan_recs, color=colors)
ax.set_ylabel('预测恢复时间 (天)')
ax.set_title('不同应急资源调配方案的恢复时间对比')
for bar, rec in zip(bars, plan_recs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{rec:.1f}天', ha='center')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()