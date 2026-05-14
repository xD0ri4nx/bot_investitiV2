# Apex Quant AI: Institutional Trading Architecture

## Overview
Apex Quant AI is an advanced, multi-stage algorithmic trading pipeline designed for high-performance execution within Google Colab environments. It leverages state-of-the-art Natural Language Processing (NLP) for market sentiment analysis and deep sequential modeling to predict asset price action and manage portfolio risk.

The architecture is built on a "Compute Hierarchy" methodology, isolating data engineering tasks to free-tier hardware and reserving high-VRAM execution (Nvidia L4/G4 GPUs) exclusively for deep hyperparameter optimization and neural network training.

## Core Architecture & Features

### 1. The Data Forge (Feature Engineering & NLP)
* **FinBERT Sentiment Miner:** Integrates Hugging Face's `ProsusAI/finbert` via pipeline batching to read, classify, and score thousands of historical financial news articles, translating raw text into quantitative sentiment tensors.
* **Advanced Market Microstructure:** Calculates complex indicators including Garman-Klass Volatility, Fractional Differentiation, Volume Price Trend, MACD Histograms, and Gaussian Mixture Model (GMM) Volatility Regimes.
* **Macro-Economic Integration:** Ingests external stress indicators (VIX, TNX) to prevent the model from trading blindly during systemic market breakdowns.

### 2. The GPU Optimizer (Optuna Deep Search)
* **VRAM-Optimized Training:** Utilizes `mixed_float16` precision to maximize batch sizes on enterprise hardware.
* **Hyperband Pruning:** Integrates Optuna with a rigorous Hyperband Pruner to aggressively terminate underperforming trials, saving compute credits.
* **Custom Objective Functions:** Optimized to maximize the Out-Of-Sample Sharpe Ratio and manage risk, rather than simply minimizing categorical cross-entropy.

### 3. Deep Learning Engine
* **BiLSTM + Multi-Head Attention:** Upgraded from standard LSTMs. The neural network utilizes Bidirectional Long Short-Term Memory layers paired with Multi-Head Self-Attention mechanisms to correlate long-term historical price action with immediate micro-structure events without suffering from memory decay.
* **Huber Loss Regression:** Predicts continuous log returns with resilience to market outliers and extreme volatility spikes.

### 4. Execution & Dashboarding
* **Institutional Shielding:** Implements rigorous risk management through strict Trailing Stop algorithms and Relative Dominance entry logic.
* **Native Colab Tunneling:** Bypasses third-party proxy services to host a local Streamlit dashboard directly within the Google Colab environment, providing real-time views of model parameters and equity curves.

## Workflow & Deployment

This project is optimized for a two-tier compute pipeline to maximize resource efficiency.

### Phase 1: Copper Execution (Data Preparation)
**Environment:** Standard CPU or Free T4 GPU
1. Configure `TICKERS` in the setup cells.
2. Run the FinBERT pipeline to generate sentiment CSVs.
3. Run the Feature Engineering block to merge Yahoo Finance data, macroeconomic indicators, and sentiment scores.
4. Save the pre-processed tensors and master DataFrames to Google Drive. Disconnect runtime.

### Phase 2: Gold Execution (Deep Optimization)
**Environment:** Paid Compute (Nvidia L4 or G4 Blackwell)
1. Mount Google Drive and load the pre-engineered master CSVs.
2. Execute the 50 to 100-trial Optuna Hyperband search.
3. Evaluate the Out-Of-Sample Walk-Forward validation metrics (Sharpe Ratio, Max Drawdown).
4. Launch the Streamlit visualization cell to open the secure native proxy tunnel.
5. Review the dashboard and terminate the runtime.

## Requirements

* Python 3.10+
* TensorFlow / Keras 3.0+
* Optuna
* Hugging Face `transformers` & `datasets`
* Streamlit
* yfinance
* scikit-learn

## Disclaimer
This software is for educational and research purposes only. It is not financial advice. Algorithmic trading carries significant financial risk. The developers of this repository assume no responsibility for any capital losses incurred while deploying these models in live markets. Always test thoroughly using paper trading before committing real capital.
