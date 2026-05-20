import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# ==================== LSTM碳价预测 + 最优交易策略（3060零碳基地）====================
class CarbonLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = CarbonLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 模拟历史数据（碳价 + 宏观变量）
np.random.seed(42)
data = np.random.normal(50, 10, (100, 5)).astype(np.float32)  # 5个特征
target = np.random.normal(55, 8, 100).astype(np.float32)

# 训练（demo用）
X = torch.from_numpy(data).unsqueeze(1)
y = torch.from_numpy(target).unsqueeze(1)
for epoch in range(50):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

# 预测未来30天
future = torch.randn(30, 1, 5)
pred_prices = model(future).detach().numpy().flatten().round(1)

print("=== 碳价预测与交易策略报告（京东方能源科技基地）===")
print(pd.DataFrame({'未来天数': range(1, 31), '预测碳价(元/吨)': pred_prices}))

# 简单策略矩阵
strategy = np.where(pred_prices > 60, '卖出', np.where(pred_prices < 45, '买入', '持有'))
print("\n最优碳配额交易策略（Top 5天）:")
print(pd.DataFrame({'天数': range(1, 6), '预测价': pred_prices[:5], '建议': strategy[:5]}))