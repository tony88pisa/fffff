#!/usr/bin/env python3
"""
Test finale semplificato del sistema HRM
Evita problemi di import relativi usando import assoluti
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Aggiungi src al path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path.absolute()))

def test_hrm_core():
    """Test del core HRM"""
    print("🧪 Test HRM Core...")
    try:
        from hrm.core import HRM, HRMConfig
        
        config = HRMConfig(
            input_dim=128,
            hidden_dim=256,
            n_high_layers=4,
            n_low_layers=2,
            n_heads=8,
            n_cycles=3,
            steps_per_cycle=10,
            max_seq_len=100
        )
        
        model = HRM(config)
        print(f"✅ HRM creato: {type(model).__name__}")
        print(f"   - Input dim: {config.input_dim}")
        print(f"   - Hidden dim: {config.hidden_dim}")
        print(f"   - Layers: {config.n_high_layers}/{config.n_low_layers}")
        return True
        
    except Exception as e:
        print(f"❌ Errore HRM Core: {e}")
        return False

def test_feature_adapter():
    """Test del FeatureAdapter"""
    print("\n🧪 Test FeatureAdapter...")
    try:
        from hrm.heads import FeatureAdapter, MarketFeatures
        
        # Usa parametri corretti basati sulla definizione
        adapter = FeatureAdapter(
            n_timeframes=4,
            n_technical_indicators=10,
            n_orderbook_features=3,
            n_portfolio_features=3,
            output_dim=128,
            sequence_length=100
        )
        
        # Test con MarketFeatures
        features = MarketFeatures(
            prices=torch.randn(4, 100),
            volumes=torch.randn(4, 100),
            returns=torch.randn(4, 100),
            volatility=torch.randn(4, 100),
            rsi=torch.randn(4, 100),
            macd=torch.randn(4, 100),
            bollinger_bands=torch.randn(4, 100, 3),
            position=torch.tensor([1000.0]),
            unrealized_pnl=torch.tensor([150.0]),
            risk_exposure=torch.tensor([0.3]),
            tick_size=0.01,
            lot_size=100,
            min_notional=1000.0,
            symbol="BTCUSDT",
            timestamp=1234567890,
            timeframes=["1m", "5m", "15m", "1h"]
        )
        
        output = adapter(features)
        print(f"✅ FeatureAdapter funziona")
        print(f"   - Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Errore FeatureAdapter: {e}")
        return False

def test_output_head():
    """Test dell'output head"""
    print("\n🧪 Test Output Head...")
    try:
        from hrm.heads import OutputHead
        
        input_dim = 256
        
        # Test OutputHead
        output_head = OutputHead(input_dim=input_dim)
        
        # Test con dati sintetici
        batch_size = 2
        seq_len = 100
        hidden_states = torch.randn(batch_size, seq_len, input_dim)
        
        output = output_head(hidden_states)
        
        print(f"✅ Output Head funziona")
        print(f"   - Output keys: {list(output.keys())}")
        for key, value in output.items():
            print(f"   - {key}: {value.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Errore Output Head: {e}")
        return False

def test_market_features():
    """Test della dataclass MarketFeatures"""
    print("\n🧪 Test MarketFeatures...")
    try:
        from hrm.heads import MarketFeatures
        
        # Crea features di mercato
        features = MarketFeatures(
            prices=torch.randn(4, 100),
            volumes=torch.randn(4, 100),
            returns=torch.randn(4, 100),
            volatility=torch.randn(4, 100),
            rsi=torch.randn(4, 100),
            macd=torch.randn(4, 100),
            bollinger_bands=torch.randn(4, 100, 3),
            position=torch.tensor([1000.0]),
            unrealized_pnl=torch.tensor([150.0]),
            risk_exposure=torch.tensor([0.3]),
            tick_size=0.01,
            lot_size=100,
            min_notional=1000.0,
            symbol="BTCUSDT",
            timestamp=1234567890,
            timeframes=["1m", "5m", "15m", "1h"]
        )
        
        print(f"✅ MarketFeatures creato")
        print(f"   - Symbol: {features.symbol}")
        print(f"   - Position: {features.position.item()}")
        print(f"   - Timeframes: {len(features.timeframes)}")
        return True
        
    except Exception as e:
        print(f"❌ Errore MarketFeatures: {e}")
        return False

def test_end_to_end():
    """Test end-to-end semplificato"""
    print("\n🧪 Test End-to-End...")
    try:
        from hrm.core import HRM, HRMConfig
        from hrm.heads import FeatureAdapter, OutputHead, MarketFeatures
        
        # Configurazione
        config = HRMConfig(
            input_dim=128,
            hidden_dim=256,
            n_high_layers=2,
            n_low_layers=1,
            n_heads=4,
            n_cycles=2,
            steps_per_cycle=5,
            max_seq_len=50
        )
        
        # Componenti
        feature_adapter = FeatureAdapter(
            n_timeframes=4,
            n_technical_indicators=10,
            n_orderbook_features=3,
            n_portfolio_features=3,
            output_dim=config.input_dim, 
            sequence_length=config.max_seq_len
        )
        hrm_model = HRM(config)
        output_head = OutputHead(input_dim=config.hidden_dim)  # Deve essere coerente con HRM output dim
        
        # Pipeline
        # 1. Crea MarketFeatures con dimensioni corrette [batch_size, seq_len, n_timeframes]
        batch_size = 4
        seq_len = config.max_seq_len
        n_timeframes = 4
        
        features = MarketFeatures(
            prices=torch.randn(batch_size, seq_len, n_timeframes),
            volumes=torch.randn(batch_size, seq_len, n_timeframes),
            returns=torch.randn(batch_size, seq_len, n_timeframes),
            volatility=torch.randn(batch_size, seq_len, n_timeframes),
            rsi=torch.randn(batch_size, seq_len, n_timeframes),
            macd=torch.randn(batch_size, seq_len, n_timeframes),
            bollinger_bands=torch.randn(batch_size, seq_len, n_timeframes, 3),
            position=torch.tensor([1000.0]),
            unrealized_pnl=torch.tensor([150.0]),
            risk_exposure=torch.tensor([0.3]),
            tick_size=0.01,
            lot_size=100,
            min_notional=1000.0,
            symbol="BTCUSDT",
            timestamp=1234567890,
            timeframes=["1m", "5m", "15m", "1h"]
        )
        
        # 2. Adatta features
        adapted_features = feature_adapter(features)
        assert isinstance(adapted_features, torch.Tensor), f"adapter must return Tensor, got {type(adapted_features)}"
        
        # 3. Processa con HRM
        hrm_output = hrm_model(adapted_features)  # return_dict=False per default
        if hrm_output is None:
            raise AssertionError("HRM core returned None. Ensure forward returns a Tensor or set return_dict and extract 'x'.")
        if isinstance(hrm_output, dict):
            hrm_output = hrm_output.get("output") or hrm_output.get("x") or hrm_output.get("hidden") or hrm_output.get("last_hidden_state")
            assert isinstance(hrm_output, torch.Tensor), f"HRM core dict missing main tensor key. Keys: {list(hrm_output.keys()) if isinstance(hrm_output, dict) else 'N/A'}"
        assert isinstance(hrm_output, torch.Tensor), f"core must return Tensor, got {type(hrm_output)}"
        assert hrm_output.ndim == 3, f"core output must be [B,S,D], got {hrm_output.shape}"
        B, S, D_core = hrm_output.shape
        assert D_core == output_head.input_dim, f"OutputHead.input_dim={output_head.input_dim} ≠ core D={D_core}"
        
        # 4. Genera decisioni trading
        trading_decisions = output_head(hrm_output)
        assert isinstance(trading_decisions, dict), f"head must return dict, got {type(trading_decisions)}"
        
        print(f"✅ Pipeline End-to-End funziona")
        print(f"   - Adapted features: {adapted_features.shape}")
        print(f"   - HRM output: {hrm_output.shape}")
        print(f"   - Trading decisions: {list(trading_decisions.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Errore End-to-End: {e}")
        return False

def main():
    """Esegue tutti i test"""
    print("=" * 60)
    print("🚀 TEST FINALE SISTEMA HRM - VERSIONE SEMPLIFICATA")
    print("=" * 60)
    
    tests = [
        ("HRM Core", test_hrm_core),
        ("FeatureAdapter", test_feature_adapter),
        ("Output Head", test_output_head),
        ("MarketFeatures", test_market_features),
        ("End-to-End", test_end_to_end)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Report finale
    print("\n" + "=" * 60)
    print("📊 REPORT FINALE")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nRisultato: {passed}/{total} test passati")
    print(f"Tasso di successo: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 Tutti i test sono passati! Sistema HRM pronto.")
    else:
        print(f"\n⚠️  {total-passed} test falliti. Rivedere i componenti.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)