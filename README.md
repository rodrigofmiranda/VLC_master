# Modelagem de Canal para VLC com VAE

Este repositório contém os notebooks e scripts do projeto de modelagem de canal para sistemas de Comunicação por Luz Visível (VLC) usando Variational Autoencoders (VAE).

## Estrutura
- `notebooks/`: Jupyter Notebooks com experimentos.
- `src/`: Scripts modulares em Python.
- `data/`: Dados brutos ou processados (não versionados).
- `results/`: Métricas, gráficos e logs.

## Como executar
1. Instale os pacotes:
   ```bash
   pip install -r requirements.txt
   ```
   Os scripts de geração de dados utilizam o GNU Radio e suas dependências
   (por exemplo `PyQt5` e módulos `gnuradio`), que não estão listadas no
   arquivo `requirements.txt`.
