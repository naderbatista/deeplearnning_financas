import requests
import pandas as pd

BYBIT_API_URL = "https://api.bybit.com/v5/market/kline"

# 1. Coleta de dados da Bybit
def dados_candles(symbol, interval, limit=1000):

    try:
        # Parâmetros para a API da Bybit
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = requests.get(BYBIT_API_URL, params=params)
        data = response.json()

        # Verifica se a API retornou erro
        if data.get("retCode") != 0:
            print(f"Erro ao buscar dados da Bybit: {data.get('retMsg')}")
            return []

        # Obtém e formata os candles
        candles = data["result"]["list"]
        formatted_candles = [
            {
                "time": int(candle[0]) // 1000,  # Converte timestamp de milissegundos para segundos
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4])
            }
            for candle in candles
        ]

        # Ordena os candles em ordem cronológica crescente
        formatted_candles.sort(key=lambda x: x["time"])

        return formatted_candles
    except Exception as e:
        print(f"Erro ao buscar dados da Bybit: {e}")
        return []

def dados_analise(symbol, interval, limit):
    try:
        # Parâmetros para a API da Bybit
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = requests.get(BYBIT_API_URL, params=params)
        data = response.json()

        # Verifica se a API retornou erro
        if data.get("retCode") != 0:
            print(f"Erro ao buscar dados da Bybit: {data.get('retMsg')}")
            return pd.DataFrame()  # Retorna um DataFrame vazio

        # Obtém e formata os candles
        candles = data["result"]["list"]
        formatted_candles = [
            {
                "time": int(candle[0]) // 1000,  # Converte timestamp de milissegundos para segundos
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])       # Volume
            }
            for candle in candles
        ]

        return pd.DataFrame(formatted_candles)  # Retorna os candles formatados como DataFrame
    except Exception as e:
        print(f"Erro ao buscar dados da Bybit: {e}")
        return pd.DataFrame()  # Retorna um DataFrame vazio em caso de falha
