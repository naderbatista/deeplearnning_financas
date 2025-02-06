# Usar uma imagem base do Python
FROM python:3.9-slim

# Instalar pacotes necessários do sistema
RUN apt-get update && apt-get install -y \
    libgomp1 libquadmath0 libopenblas-dev liblapack-dev libatlas-base-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos do projeto para dentro do container
COPY . .

# Instalar as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Corrigir erro de 'np.bool' automaticamente no MXNet
RUN sed -i "s/onp\.bool\b/bool/g" /usr/local/lib/python3.9/site-packages/mxnet/numpy/utils.py

# Definir o comando padrão para rodar a aplicação
CMD ["python", "server.py"]
