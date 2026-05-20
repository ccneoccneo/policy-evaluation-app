"""
碳价预测与配额交易策略（PyTorch LSTM）
功能：使用历史碳价训练LSTM模型（PyTorch），预测未来60个交易日碳价，生成买卖建议
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# ==================== 1. 生成模拟碳价数据 ====================
np.random.seed(42)
torch.manual_seed(42)
n_days = 500
dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

carbon_prices = [80]
for i in range(1, n_days):
    ar = 0.9 * (carbon_prices[-1] - 80)
    noise = np.random.normal(0, 2.5)
    new_price = 80 + ar + noise
    carbon_prices.append(max(30, new_price))

df = pd.DataFrame({'date': dates, 'carbon_price': carbon_prices}).set_index('date')

print("=" * 50)
print("碳市场数据概况")
print("=" * 50)
print(f"数据周期: {df.index.min().date()} 至 {df.index.max().date()}")
print(f"交易日数: {len(df)}")
print(f"碳价均值: {df['carbon_price'].mean():.2f} 元/吨")
print(f"碳价标准差: {df['carbon_price'].std():.2f}")

# ==================== 2. 数据预处理 ====================
prices = df['carbon_price'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(prices_scaled, seq_length)
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(-1, 1)

split = len(X) - 120
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 转换为PyTorch张量
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==================== 3. 定义LSTM模型 ====================
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)          # [batch, seq, hidden]
        last_out = lstm_out[:, -1, :]       # 取最后时间步
        out = self.relu(self.fc1(last_out))
        out = self.fc2(out)
        return out

model = LSTMPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
epochs = 30
print("\n开始训练LSTM模型 (PyTorch)...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train):.6f}")
print("训练完成。")

# 测试集预测
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).numpy()
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_inv = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))
print(f"测试集RMSE: {rmse:.2f} 元/吨")

# ==================== 4. 预测未来60个交易日 ====================
model.eval()
last_seq = prices_scaled[-seq_length:].reshape(1, seq_length, 1)
last_seq_t = torch.tensor(last_seq, dtype=torch.float32)

future_preds_scaled = []
current_seq = last_seq_t.clone()
with torch.no_grad():
    for _ in range(60):
        next_val = model(current_seq).numpy()[0, 0]
        future_preds_scaled.append(next_val)
        # 更新序列
        new_val = torch.tensor([[[next_val]]], dtype=torch.float32)
        current_seq = torch.cat([current_seq[:, 1:, :], new_val], dim=1)

future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=60, freq='B')

# ==================== 5. 企业配额与交易策略 ====================
annual_emission = 5000000
free_allowance = 4500000
quota_gap = annual_emission - free_allowance

current_price = df['carbon_price'].iloc[-1]
pred_30d_avg = future_preds[:30].mean()
pred_60d_avg = future_preds.mean()

print("\n" + "=" * 50)
print("企业碳资产管理概览")
print("=" * 50)
print(f"年度排放量: {annual_emission:,} 吨")
print(f"免费配额: {free_allowance:,} 吨")
print(f"配额缺口: {quota_gap:,} 吨 {'(需买入)' if quota_gap > 0 else '(有盈余可卖出)'}")
print(f"当前碳价: {current_price:.2f} 元/吨")
print(f"预测未来30日均价: {pred_30d_avg:.2f} 元/吨")
print(f"预测未来60日均价: {pred_60d_avg:.2f} 元/吨")

print("\n--- 交易策略建议 ---")
if quota_gap > 0:
    if pred_30d_avg > current_price * 1.03:
        print(f"【立即买入】约 {quota_gap * 0.8:.0f} 吨")
        print(f"理由: 预测30日均价上涨超过3%，建议锁定成本")
    elif pred_60d_avg < current_price * 0.97:
        print(f"【暂缓买入】等待价格回落")
        print(f"理由: 预测60日均价下跌，延迟买入更划算")
    else:
        print(f"【分批买入】分3个月逐步建仓，每月约{quota_gap/3:.0f}吨")
else:
    if pred_30d_avg < current_price * 0.98:
        print(f"【立即卖出】约 {abs(quota_gap) * 0.7:.0f} 吨")
    elif pred_60d_avg > current_price * 1.05:
        print(f"【暂持待涨】等待更高价位")
    else:
        print(f"【分批卖出】每季度卖出盈余的25%")

# ==================== 6. 可视化 ====================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax1 = axes[0]
ax1.plot(df.index[-200:], df['carbon_price'].iloc[-200:], color='steelblue', label='历史碳价')
ax1.plot(future_dates, future_preds, 'r--', label='LSTM预测')
ax1.fill_between(future_dates, (future_preds - rmse).flatten(), (future_preds + rmse).flatten(),
                 color='red', alpha=0.15, label='预测区间(±RMSE)')
ax1.axhline(y=current_price, color='gray', linestyle=':', alpha=0.7)
ax1.set_ylabel('碳价 (元/吨)')
ax1.set_title('碳价历史走势与未来60日预测 (PyTorch LSTM)')
ax1.legend(); ax1.grid(alpha=0.3)

ax2 = axes[1]
test_dates = df.index[-len(y_test):]
ax2.plot(test_dates, y_test_inv, label='真实值')
ax2.plot(test_dates, y_pred, '--', label='预测值')
ax2.set_ylabel('碳价'); ax2.set_title(f'测试集表现 (RMSE={rmse:.2f})')
ax2.legend(); ax2.grid(alpha=0.3)

ax3 = axes[2]
ax3.hist(future_preds, bins=20, color='coral', edgecolor='white')
ax3.axvline(current_price, color='blue', linestyle='--', label=f'当前价格 {current_price:.1f}')
ax3.axvline(pred_30d_avg, color='red', linestyle='--', label=f'30日均价 {pred_30d_avg:.1f}')
ax3.set_xlabel('碳价'); ax3.set_title('未来60日碳价预测分布')
ax3.legend(); ax3.grid(alpha=0.3)
plt.tight_layout()
plt.show()