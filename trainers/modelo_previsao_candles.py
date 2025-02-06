import mxnet as mx
from mxnet import gluon, autograd, nd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os
import ta
import requests
import sys
import matplotlib.pyplot as plt
from config.corretoras import dados_candles

# Configuração para CPU
ctx = mx.cpu()

# 2. Adicionar indicadores técnicos
def adiciona_indicadores(df):
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['vol_sma'] = ta.trend.SMAIndicator(close=df['volume'], window=10).sma_indicator()
    df.fillna(0, inplace=True)
    return df

# 3. Preparar os dados
def prepara_dados(df, seq_length=50, future_steps=20):
    df = adiciona_indicadores(df)
    features = ['close', 'rsi', 'macd', 'ema', 'vol_sma']
    data = df[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)

    X, Y = [], []
    for i in range(len(data_normalized) - seq_length - future_steps):
        X.append(data_normalized[i:i + seq_length])
        Y.append(data_normalized[i + seq_length:i + seq_length + future_steps, 0])

    X = np.array(X).reshape(-1, seq_length, len(features))
    Y = np.array(Y).reshape(-1, future_steps)

    return nd.array(X, ctx=ctx), nd.array(Y, ctx=ctx), scaler

# 4. Criar o modelo LSTM
class LSTMModel(gluon.Block):
    def __init__(self, hidden_size=128, num_layers=3, future_steps=20, **kwargs):
        super(LSTMModel, self).__init__(**kwargs)
        self.lstm = gluon.rnn.LSTM(hidden_size, num_layers=num_layers, layout='NTC')
        self.dense = gluon.nn.Dense(future_steps)

    def forward(self, x):
        x = self.lstm(x)
        return self.dense(x[:, -1, :])

# 5. Treinar o modelo
def treina_modelo(model, X_train, Y_train, epochs=50, batch_size=32, lr=0.001):
    dataset = gluon.data.ArrayDataset(X_train, Y_train)
    dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.initialize(mx.init.Xavier(), ctx=ctx)
    loss_fn = gluon.loss.L2Loss()
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})

    print(f"Treinando o modelo para {epochs} épocas...")

    for epoch in range(epochs):
        cumulative_loss = 0
        for X_batch, Y_batch in dataloader:
            with autograd.record():
                Y_pred = model(X_batch)
                loss = loss_fn(Y_pred, Y_batch)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += loss.mean().asscalar()

        print(f"Epoch {epoch + 1}, Loss: {cumulative_loss / len(dataloader)}")

    print("Treinamento concluído!")
    # Salvar o modelo treinado
    path = f"models/{symbol}_{interval}"
    os.makedirs(path, exist_ok=True)
    model.save_parameters(f"{path}/modelo_lstm.params")
    print(f"Modelo salvo em {path}/modelo_lstm.params")

# 6. Backtest e plotagem
def backtest(model, X_test, Y_test, scaler):
    Y_pred = model(X_test).asnumpy()

    # Extraímos apenas os valores de fechamento da normalização
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]  # Usamos apenas os valores da coluna 'close'

    # Reverter normalização apenas para os preços de fechamento
    Y_test_real = close_scaler.inverse_transform(Y_test.asnumpy())
    Y_pred_real = close_scaler.inverse_transform(Y_pred)

    # Calcular métricas para cada previsão futura
    for i in range(Y_test_real.shape[1]):
        mae = mean_absolute_error(Y_test_real[:, i], Y_pred_real[:, i])
        mse = mean_squared_error(Y_test_real[:, i], Y_pred_real[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test_real[:, i], Y_pred_real[:, i])
        print(f"🔹 Candle +{i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    plota_resultados(Y_test_real[:, 0], Y_pred_real[:, 0])

def plota_resultados(y_test, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Preço Real", color="blue")
    plt.plot(predictions, label="Previsão", color="red", linestyle="dashed")
    plt.xlabel("Tempo")
    plt.ylabel("Preço")
    plt.title("Projeção dos Próximos Candles")
    plt.legend()

    output_dir = "/resultado_backtest"
    output_path = f"{output_dir}/previsao_lstm.png"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# 7. Fluxo principal
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso correto: python modelo.py SYMBOL INTERVAL")
        sys.exit(1)

    symbol = sys.argv[1]
    interval = sys.argv[2]

    print(f"Baixando dados para {symbol} no intervalo {interval}...")
    df = dados_candles(symbol, interval)
    if df.empty:
        print("Erro ao buscar os dados. Verifique o par de negociação e o intervalo.")
        sys.exit(1)

    print("Preparando os dados...")
    X_train, Y_train, scaler = prepara_dados(df)
    split = int(len(X_train) * 0.8)
    X_test, Y_test = X_train[split:], Y_train[split:]

    print("Criando e treinando o modelo...")
    model = LSTMModel(future_steps=20)
    treina_modelo(model, X_train, Y_train)

    print("Executando backtest...")
    backtest(model, X_test, Y_test, scaler)
