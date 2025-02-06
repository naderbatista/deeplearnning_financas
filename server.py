from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import mxnet as mx
from mxnet import gluon, nd
import numpy as np
import os
import time
from sklearn.preprocessing import MinMaxScaler
from threading import Thread, Lock
from trainers.modelo_previsao_candles import LSTMModel, prepara_dados, treina_modelo
from config.corretoras import dados_candles,dados_analise

# Configuração do Flask
app = Flask(__name__, static_folder="static", template_folder="pages")
CORS(app)

# Configuração do modelo
ctx = mx.cpu()
MODEL_DIR = "models"
SEQ_LENGTH, FUTURE_STEPS = 50, 20
FEATURES = ["close", "rsi", "macd", "ema", "vol_sma"]

# Controle de Treinamento
training_lock = Lock()
training_status = {}

@app.route("/")
def index():
    return render_template("index.html")

# Verifica se o modelo já foi salvo
def modelo_salvo(symbol, interval):
    return os.path.exists(f"{MODEL_DIR}/{symbol}_{interval}/modelo_lstm.params")

# Treina o modelo em background
def fila_treinamento(symbol, interval):
    with training_lock:
        if modelo_salvo(symbol, interval):
            print(f"Modelo {symbol} ({interval}) já existe.")
            return
        
        training_status[(symbol, interval)] = True
        print(f"Treinando modelo para {symbol} ({interval})...")
        
        df = dados_candles(symbol, interval)
        if df.empty:
            print(f"Erro: Sem dados para {symbol} ({interval}).")
            training_status[(symbol, interval)] = False
            return
        
        X_train, Y_train, scaler = prepara_dados(df)
        model = LSTMModel(future_steps=FUTURE_STEPS)
        treina_modelo(model, X_train, Y_train, symbol, interval)
        
        training_status[(symbol, interval)] = False

# Carrega ou treina o modelo se necessário
def carrega_ou_treina(symbol, interval):
    if modelo_salvo(symbol, interval):
        model = LSTMModel(future_steps=FUTURE_STEPS)
        model.load_parameters(f"{MODEL_DIR}/{symbol}_{interval}/modelo_lstm.params", ctx=ctx)
        print(f"Modelo {symbol} ({interval}) carregado!")
        return model

    print(f"⚠️ Modelo {symbol} ({interval}) não encontrado. Treine o modelo antes de usá-lo.")
    return None

@app.route("/candles", methods=["GET"])
def carrega_candles():
    symbol = request.args.get("symbol")
    interval = request.args.get("interval")
    if not symbol or not interval:
        return jsonify({"error": "Parâmetros 'symbol' e 'interval' são obrigatórios"}), 400
    candles = dados_candles(symbol, interval)
    if not candles:
        return jsonify({"error": "Nenhum dado disponível."}), 400
    return jsonify(candles)

@app.route("/realtime", methods=["GET"])
def realtime():
    """Retorna previsões futuras com base nos últimos candles coletados."""
    try:
        symbol = request.args.get("symbol")
        interval = request.args.get("interval")

        if not symbol or not interval:
            return jsonify({"error": "Parâmetros 'symbol' e 'interval' são obrigatórios."}), 400

        # Definir o limite de candles como 1000
        limit = 500  

        # Carregar ou treinar o modelo
        model = carrega_ou_treina(symbol, interval)
        if model is None:
            return jsonify({"message": f"⏳ Treinamento de {symbol} ({interval}) em andamento..."}), 202

        # Obter dados de análise (usando 1000 candles)
        df = dados_analise(symbol, interval, limit=limit)
        if df.empty or len(df) < SEQ_LENGTH: 
            return jsonify({"error": "Dados insuficientes para prever."}), 500

        # Converter os últimos 50 candles para uma lista de dicionários
        real_candles = df.tail(50).to_dict(orient="records")

        # Certificar-se de que cada candle tem um campo "time"
        if not all("time" in candle for candle in real_candles):
            return jsonify({"error": "Campo 'time' ausente nos candles reais."}), 500

        # Converter timestamps para milissegundos
        for candle in real_candles:
            candle["time"] = int(candle["time"]) * 1000 

        # Obter o último timestamp real em milissegundos
        last_time_real = int(df["time"].iloc[-1]) * 1000  

        # Obter o tempo atual do sistema
        current_time = int(time.time() * 1000)

        # Se o último timestamp real estiver muito atrás do tempo atual, ajustamos a previsão para iniciar do presente
        if last_time_real < current_time - (int(interval) * 60 * 1000):
            last_time = current_time  
        else:
            last_time = last_time_real

        # Preparar dados para o modelo
        X_test, _, scaler = prepara_dados(df)

        # Fazer previsão com os últimos 1000 candles
        Y_pred = model(X_test[-1].reshape(1, SEQ_LENGTH, len(FEATURES))).asnumpy()

        # Reverter normalização apenas para os preços de fechamento
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
        Y_pred_real = close_scaler.inverse_transform(Y_pred).flatten()

        # Gerar os candles previstos corretamente no futuro
        previsao_candles = []
        for i, value in enumerate(Y_pred_real):
            future_time = last_time + ((i + 1) * int(interval) * 60 * 1000)
            
            # Garantir que o `future_time` está sempre avançando corretamente
            if future_time < current_time:
                future_time = current_time + ((i + 1) * int(interval) * 60 * 1000)

            previsao_candles.append({
                "t": future_time,  
                "o": float(value * 0.99),
                "h": float(value * 1.01), 
                "l": float(value * 0.98), 
                "c": float(value) 
            })

        return jsonify({
            "symbol": symbol,
            "interval": interval,
            "ohlc": real_candles,  # Candles reais formatados corretamente
            "predicted": previsao_candles  # Candles previstos com timestamps corrigidos
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
