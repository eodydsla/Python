import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import ta

# GPU 설정
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 셀트리온 주식 데이터 불러오기
df = yf.download('068270.KS', start='2010-01-01', end='2024-07-02')
df = df[['Close', 'Volume']]

# RSI 추가
df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

# 결측값 제거
df.dropna(inplace=True)

# 데이터 전처리
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close', 'Volume', 'RSI']])

# 시계열 데이터 준비
def prepare_multivariate_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 0])  # 주가 예측
    return np.array(X), np.array(y)

n_steps = 60  # 다시 60으로 설정
X, y = prepare_multivariate_data(scaled_data, n_steps)

# 데이터셋 분할
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.2)
test_size = len(X) - train_size - val_size

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

# DataLoader 생성
batch_size = 64

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 트랜스포머 모델 정의
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, n_steps, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x

model = TransformerTimeSeries(input_dim=3, model_dim=64, num_heads=4, num_layers=3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 조기 중지 클래스 정의
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict()

early_stopping = EarlyStopping(patience=10, min_delta=0)

# 모델 훈련
n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_output = model(X_batch)
            loss = criterion(val_output, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# 최종 모델 저장
torch.save(early_stopping.best_model, 'transformer_timeseries_model_best.pth')

# 모델 불러오기
model.load_state_dict(torch.load('transformer_timeseries_model_best.pth'))
model.to(device)
model.eval()

# 예측 함수 정의
def predict_future(model, data, steps):
    model.eval()
    predictions = []
    current_data = data.copy()
    
    for _ in range(steps):
        with torch.no_grad():
            input_data = torch.tensor(current_data[-n_steps:], dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(input_data).cpu().numpy().flatten()[0]
            predictions.append(pred)
            current_data = np.append(current_data, [[pred, 0, 0]], axis=0)
            current_data = current_data[1:]
    
    return predictions

# 모델 평가 및 예측
predicted = []
actual = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(X_batch).squeeze().cpu().numpy()
        predicted.extend(pred)
        actual.extend(y_batch.squeeze().cpu().numpy())

future_steps = 30
scaled_last_data = scaled_data[-n_steps:]
future_predictions = predict_future(model, scaled_last_data, future_steps)

# 예측값 반전처리
predicted = np.array(predicted)
actual = np.array(actual)

predicted = scaler.inverse_transform(np.concatenate((predicted.reshape(-1, 1), np.zeros((predicted.shape[0], 2))), axis=1))[:, 0]
actual = scaler.inverse_transform(np.concatenate((actual.reshape(-1, 1), np.zeros((actual.shape[0], 2))), axis=1))[:, 0]
future_predictions = np.array(future_predictions)
future_predictions = scaler.inverse_transform(np.concatenate((future_predictions.reshape(-1, 1), np.zeros((future_predictions.shape[0], 2))), axis=1))[:, 0]

# 결과 시각화
future_dates = pd.date_range(start=df.index[-1], periods=future_steps+1, inclusive='right')
plt.figure(figsize=(10, 6))
plt.plot(df.index[-test_size:], actual, label='Actual Close Price')
plt.plot(df.index[-test_size:], predicted, label='Predicted Close Price')
plt.plot(future_dates, future_predictions, label='Future Predictions')

# 마지막 예측 가격 표시
plt.scatter(future_dates[-1], future_predictions[-1], color='red', zorder=5)
plt.text(future_dates[-1], future_predictions[-1], f'{future_predictions[-1]:.2f}', color='red', fontsize=12, ha='right')
