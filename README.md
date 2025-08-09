# 🛡️ Guardian Trading System

> Un sistema di trading algoritmico avanzato con intelligenza artificiale, gestione del rischio e memoria episodica.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](docker-compose.yml)
[![AI](https://img.shields.io/badge/AI-DeepSeek--R1-purple.svg)](https://deepseek.com)

## 🎯 Panoramica

Guardian Trading System è un sistema di trading algoritmico di nuova generazione che combina:

- **🧠 Intelligenza Artificiale**: Modelli LLM DeepSeek-R1 per analisi di mercato avanzate
- **🛡️ Risk Management**: Sistema di controllo del rischio multi-livello
- **💾 Memoria Episodica**: Database vettoriale Weaviate per apprendimento continuo
- **📊 Portfolio Management**: Gestione completa del portafoglio con metriche avanzate
- **⚡ Real-time Processing**: Elaborazione dati in tempo reale con caching Redis
- **📈 Strategy Framework**: Framework modulare per strategie di trading personalizzate

## 🏗️ Architettura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    Guardian Trading System                      │
├─────────────────────────────────────────────────────────────────┤
│  🧠 AI Layer (DeepSeek-R1)  │  🛡️ Risk Guardian Agent         │
│  • Market Analysis           │  • Real-time Risk Assessment     │
│  • Signal Generation         │  • Position Sizing               │
│  • Strategy Optimization     │  • Drawdown Protection           │
├─────────────────────────────────────────────────────────────────┤
│  💾 Memory System           │  📊 Portfolio Manager            │
│  • Weaviate Vector DB       │  • Position Tracking             │
│  • Trade Memories           │  • Performance Analytics         │
│  • Market Reflections       │  • Order Management              │
├─────────────────────────────────────────────────────────────────┤
│  📈 Strategy Manager        │  📡 Data Manager                 │
│  • MA Cross Strategy         │  • Multi-source Data Feed        │
│  • RSI Strategy              │  • Data Validation               │
│  • Bollinger Bands          │  • Real-time Streaming           │
├─────────────────────────────────────────────────────────────────┤
│  🔧 Infrastructure Layer                                        │
│  • Docker Compose  • Redis Cache  • PostgreSQL  • Monitoring   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Caratteristiche Principali

### 🧠 Sistema di Intelligenza Artificiale
- **Modelli LLM Avanzati**: DeepSeek-R1-Distill-Qwen-14B per analisi complesse
- **Analisi Multi-timeframe**: Elaborazione simultanea di diversi orizzonti temporali
- **Sentiment Analysis**: Analisi del sentiment di mercato in tempo reale
- **Pattern Recognition**: Riconoscimento automatico di pattern di trading

### 🛡️ Risk Guardian Agent
- **Controllo Rischio Real-time**: Valutazione continua del rischio di portafoglio
- **Stop Loss Dinamici**: Gestione automatica degli stop loss
- **Position Sizing**: Calcolo ottimale della dimensione delle posizioni
- **Correlation Analysis**: Analisi delle correlazioni tra asset
- **VaR Calculation**: Calcolo del Value at Risk con diversi livelli di confidenza

### 💾 Sistema di Memoria Episodica
- **Vector Database**: Weaviate per memorizzazione di esperienze di trading
- **Trade Memories**: Registrazione dettagliata di ogni operazione
- **Market Reflections**: Analisi retrospettive delle condizioni di mercato
- **Continuous Learning**: Apprendimento continuo dalle esperienze passate

### 📊 Portfolio Management
- **Multi-asset Support**: Supporto per azioni, crypto, forex, commodities
- **Performance Tracking**: Metriche avanzate di performance (Sharpe, Sortino, Calmar)
- **Rebalancing Automatico**: Ribilanciamento automatico del portafoglio
- **Risk Metrics**: Calcolo di metriche di rischio avanzate

### 📈 Framework Strategie
- **Strategie Pre-built**: MA Cross, RSI, Bollinger Bands, Mean Reversion
- **Custom Strategies**: Framework per sviluppare strategie personalizzate
- **Backtesting Engine**: Sistema di backtesting con walk-forward analysis
- **Strategy Optimization**: Ottimizzazione automatica dei parametri

## Requisiti di Sistema

### Hardware Raccomandato
- **GPU**: NVIDIA RTX 5080 (o superiore)
- **CPU**: AMD Ryzen 7 3800X (o equivalente)
- **RAM**: 16GB DDR4 (minimo)
- **Storage**: 500GB SSD

### Software
- **OS**: Windows 11
- **Python**: 3.9+
- **Ollama**: Per i modelli AI locali

## Installazione

### 1. Clona il Repository
```bash
git clone <repository-url>
cd guardian
```

### 2. Installa le Dipendenze
```bash
pip install -r requirements.txt
```

### 3. Installa Ollama
1. Scarica Ollama da [ollama.ai](https://ollama.ai)
2. Installa seguendo le istruzioni per Windows
3. Verifica l'installazione:
   ```bash
   ollama --version
   ```

### 4. Configura i Modelli AI
```bash
# Installa i modelli necessari
ollama pull llama2
ollama pull codellama
ollama pull mistral
```

### 5. Configura le API Keys
Crea un file `.env` nella directory principale:
```env
# API Keys per i dati di mercato
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# Configurazioni database
DATABASE_URL=sqlite:///guardian.db

# Configurazioni sicurezza
SECRET_KEY=your_secret_key_here
```

## Avvio del Sistema

### Modalità Sviluppo
```bash
streamlit run gui/main_dashboard.py
```

### Modalità Produzione
```bash
streamlit run gui/main_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

## Struttura del Progetto

```
guardian/
├── gui/                    # Interfaccia utente Streamlit
│   ├── components/         # Componenti riutilizzabili
│   │   ├── metrics.py      # Metriche e KPI
│   │   └── agents.py       # Gestione agenti
│   ├── pages/              # Pagine dell'applicazione
│   │   ├── config.py       # Configurazioni
│   │   ├── monitoring.py   # Monitoraggio
│   │   └── trading.py      # Trading interface
│   ├── utils/              # Utilità
│   │   ├── data_manager.py # Gestione dati
│   │   └── agent_manager.py# Gestione agenti AI
│   └── main_dashboard.py   # Dashboard principale
├── agents/                 # Agenti AI (da implementare)
├── data/                   # Dati e database
├── config/                 # File di configurazione
├── logs/                   # File di log
├── requirements.txt        # Dipendenze Python
└── README.md              # Questo file
```

## Configurazione

### Agenti AI
Gli agenti possono essere configurati tramite l'interfaccia web o modificando i file di configurazione:

```python
# Esempio configurazione Market Analyst
config = AgentConfig(
    name="Market Analyst",
    agent_type=AgentType.MARKET_ANALYST,
    enabled=True,
    auto_restart=True,
    config_params={
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'analysis_interval': 300,  # 5 minuti
        'indicators': ['RSI', 'MACD', 'BB']
    }
)
```

### Gestione del Rischio
```python
# Configurazione Risk Manager
risk_config = {
    'max_position_size': 0.1,      # 10% del portfolio
    'var_limit': 0.02,             # 2% VaR giornaliero
    'max_drawdown': 0.15,          # 15% drawdown massimo
    'correlation_limit': 0.8       # Correlazione massima
}
```

## Utilizzo

### 1. Dashboard Principale
- Visualizza le performance in tempo reale
- Monitora lo stato degli agenti
- Controlla le metriche di rischio

### 2. Configurazione Agenti
- Abilita/disabilita agenti specifici
- Modifica parametri di configurazione
- Monitora i log degli agenti

### 3. Monitoraggio
- Visualizza grafici di performance
- Analizza le metriche di rischio
- Controlla lo stato del sistema

### 4. Trading
- Visualizza opportunità di trading
- Gestisci ordini e posizioni
- Analizza il portfolio

## API e Integrazioni

### Broker Supportati
- Interactive Brokers (TWS API)
- Alpaca Markets
- TD Ameritrade
- Binance (Crypto)

### Fonti Dati
- Yahoo Finance
- Alpha Vantage
- Finnhub
- Quandl

## Sicurezza

### Misure di Sicurezza
- Crittografia delle API keys
- Autenticazione multi-fattore
- Controlli di accesso basati su ruoli
- Audit trail completo

### Backup e Recovery
- Backup automatico delle configurazioni
- Snapshot del database
- Recovery point in caso di errori

## Monitoraggio e Logging

### Livelli di Log
- **DEBUG**: Informazioni dettagliate per il debugging
- **INFO**: Operazioni normali del sistema
- **WARNING**: Situazioni che richiedono attenzione
- **ERROR**: Errori che non fermano il sistema
- **CRITICAL**: Errori critici che richiedono intervento immediato

### Metriche Monitorate
- Performance degli agenti
- Utilizzo risorse di sistema
- Latenza delle operazioni
- Tasso di successo delle operazioni

## Troubleshooting

### Problemi Comuni

#### Ollama non riconosciuto
```bash
# Verifica installazione
ollama --version

# Se non funziona, aggiungi al PATH
set PATH=%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Ollama
```

#### Errori di connessione API
- Verifica le API keys nel file `.env`
- Controlla la connessione internet
- Verifica i limiti di rate delle API

#### Agenti che non si avviano
- Controlla i log degli agenti
- Verifica le configurazioni
- Riavvia il sistema se necessario

## Sviluppo

### Aggiungere Nuovi Agenti
1. Crea una nuova classe che eredita da `BaseAgent`
2. Implementa il metodo `execute()`
3. Registra l'agente nel `AgentManager`

### Aggiungere Nuove Metriche
1. Modifica la classe `MetricsComponent`
2. Aggiungi i calcoli necessari
3. Aggiorna l'interfaccia utente

### Testing
```bash
# Esegui i test
pytest tests/

# Test con coverage
pytest --cov=gui tests/
```

## Contribuire

1. Fork del repository
2. Crea un branch per la feature (`git checkout -b feature/AmazingFeature`)
3. Commit delle modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## Licenza

Questo progetto è distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori informazioni.

## Supporto

Per supporto e domande:
- Apri un issue su GitHub
- Consulta la documentazione
- Controlla i log del sistema

## Roadmap

### Versione 1.1
- [ ] Integrazione con più broker
- [ ] Modelli AI personalizzati
- [ ] Backtesting avanzato

### Versione 1.2
- [ ] Trading di criptovalute
- [ ] Analisi sentiment social media
- [ ] API REST per integrazioni esterne

### Versione 2.0
- [ ] Interfaccia mobile
- [ ] Cloud deployment
- [ ] Multi-tenancy

---

**Disclaimer**: Questo software è fornito a scopo educativo e di ricerca. Il trading comporta rischi significativi e può risultare in perdite finanziarie. Utilizzare sempre con cautela e mai con fondi che non ci si può permettere di perdere.