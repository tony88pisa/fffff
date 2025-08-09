#!/usr/bin/env python3
"""
Script di inizializzazione per il Guardian Trading System
Questo script avvia i moduli principali del sistema di trading
"""

import sys
import os
import logging
from datetime import datetime

# Aggiungi il percorso src al PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from risk_guardian import RiskGuardianAgent
    from portfolio_manager import PortfolioManager
    from data_manager import DataManager
    from strategy_manager import StrategyManager
except ImportError as e:
    print(f"Errore nell'importazione dei moduli: {e}")
    print("Assicurati che tutti i moduli siano presenti nella directory src/")
    sys.exit(1)

def setup_logging():
    """Configura il sistema di logging"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'trading_system.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('GuardianTrading')

def initialize_system():
    """Inizializza tutti i componenti del sistema"""
    logger = setup_logging()
    logger.info("🚀 Inizializzazione Guardian Trading System...")
    
    try:
        # Inizializza i componenti
        logger.info("📊 Inizializzazione Data Manager...")
        data_manager = DataManager()
        
        logger.info("💼 Inizializzazione Portfolio Manager...")
        portfolio_manager = PortfolioManager(initial_cash=100000.0)
        
        logger.info("⚖️ Inizializzazione Risk Guardian...")
        risk_guardian = RiskGuardianAgent()
        
        logger.info("🎯 Inizializzazione Strategy Manager...")
        strategy_manager = StrategyManager()
        
        # Test di connettività
        logger.info("🔍 Test dei moduli...")
        
        # Test Portfolio Manager
        metrics = portfolio_manager.calculate_metrics()
        logger.info(f"💰 Valore portfolio iniziale: ${metrics.total_value:.2f}")
        
        # Test Risk Guardian
        summary = risk_guardian.get_portfolio_summary()
        logger.info(f"⚖️ Valore portfolio: ${summary['total_value']:,.2f}")
        
        # Test Data Manager
        logger.info("📈 Test connessione dati di mercato...")
        # Qui potresti aggiungere un test di connessione ai dati reali
        
        logger.info("✅ Sistema inizializzato con successo!")
        logger.info("🌐 La GUI può ora accedere ai dati reali")
        logger.info("📱 Avvia la dashboard con: streamlit run gui/main_dashboard.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Errore durante l'inizializzazione: {e}")
        return False

if __name__ == "__main__":
    print("Guardian Trading System - Inizializzazione")
    print("="*50)
    
    success = initialize_system()
    
    if success:
        print("\n✅ Sistema pronto!")
        print("Ora puoi utilizzare la GUI con dati reali.")
    else:
        print("\n❌ Inizializzazione fallita.")
        print("Controlla i log per maggiori dettagli.")
        sys.exit(1)