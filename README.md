📊 Previsão Financeira com Deep Learning: Conheça o Projeto! 🤖

🧠 O que é Deep Learning?
Deep Learning é uma subárea do Machine Learning que utiliza redes neurais profundas para modelar e aprender padrões complexos em grandes volumes de dados. 
A principal característica do Deep Learning é sua capacidade de encontrar relações não-lineares nos dados e generalizar para cenários desconhecidos.

No caso deste projeto, o uso de uma rede LSTM é essencial, pois:

📈 É ideal para dados sequenciais e séries temporais.
🧩 Consegue capturar dependências de longo prazo nos dados financeiros.
🛠️ Reduz a perda de informações importantes durante o treinamento.

A organização do projeto segue uma estrutura limpa e modular:
=============================================================
📂 config  
   └── corretoras.py  
📂 pages  
   └── index.html  
📂 models
   └── relacao de modelos treinados
📂 trainers  
   └── modelo_previsao_candles.py  
docker-compose.yml  
Dockerfile  
requirements.txt  
server.py  
=============================================================

# O modelo é treinado com o comando:
python modelo_previsao_candles.py SYMBOL INTERVAL

=============================================================

# Implementação do Modelo
O modelo foi desenvolvido utilizando o framework MXNet/Gluon, conhecido por sua eficiência e flexibilidade em aprendizado profundo.

# Principais etapas:

Pré-processamento:
Aplicação de indicadores técnicos como RSI, MACD e EMA.
Normalização dos dados para o intervalo [0,1].

Modelo LSTM:
3 camadas com 128 unidades ocultas.
Função de perda: L2 Loss.
Otimizador: Adam.

Treinamento:
50 épocas com mini-batches de 32 amostras.
Resultados salvos automaticamente para uso posterior.

# Backtest do Modelo
Após o treinamento, realizamos backtests para avaliar a performance do modelo:

Exemplo de resultados:
🔹 Candle +1: MAE=363.5583, RMSE=489.6414, R²=0.9054  
🔹 Candle +5: MAE=473.8783, RMSE=648.7809, R²=0.8226  
🔹 Candle +10: MAE=464.0949, RMSE=663.6241, R²=0.7958  
🔹 Candle +20: MAE=604.2736, RMSE=759.1986, R²=0.6544  

Esses números mostram o potencial do modelo em prever movimentos futuros no mercado financeiro, com boa precisão em horizontes mais curtos.

💡 Participe do Projeto!
Este projeto é uma oportunidade para aprendermos e evoluirmos juntos no campo da previsão financeira com inteligência artificial.
