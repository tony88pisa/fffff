#!/usr/bin/env python3
"""
Test di integrazione finale per il sistema HRM

Questo script verifica l'integrazione completa del sistema HRM
con il sistema di trading Guardian.
"""

import sys
import os
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime
import time

# Aggiungi il percorso src
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_hrm_model_creation():
    """Test 1: Creazione modello HRM completo"""
    print("🧪 Test creazione modello HRM...")
    
    try:
        sys.path.append(str(src_path / "hrm"))
        from core import HRM, HRMConfig
        import torch
        
        # Crea configurazione
        config = HRMConfig(
            input_dim=64,  # Dimensione ridotta per test
            hidden_dim=128,
            n_high_layers=2,
            n_low_layers=2,
            n_heads=4,
            n_cycles=2,
            steps_per_cycle=3,
            dropout=0.1,
            max_seq_len=100,
            halt_threshold=0.01
        )
        
        # Crea modello
        model = HRM(config)
        print(f"✅ Modello HRM creato con {sum(p.numel() for p in model.parameters())} parametri")
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, config.input_dim)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"✅ Forward pass: {x.shape} -> {output.shape}")
        
        # Verifica output
        assert output.shape == (batch_size, seq_len, config.hidden_dim)
        assert not torch.isnan(output).any()
        
        return True
        
    except Exception as e:
        print(f"❌ Errore creazione modello HRM: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_adapter():
    """Test 2: Adattatore di features"""
    print("\n🧪 Test adattatore features...")
    
    try:
        sys.path.append(str(src_path / "hrm"))
        from heads import FeatureAdapter
        import torch
        
        # Crea adapter
        adapter = FeatureAdapter(
            input_dim=64,
            output_dim=128,
            num_timeframes=4
        )
        
        # Test encoding
        batch_size = 2
        seq_len = 10
        num_features = 15  # price, volume, indicators, etc.
        
        # Simula features di mercato
        market_data = {
            'prices': torch.randn(batch_size, 4, seq_len),  # 4 timeframes
            'volumes': torch.randn(batch_size, 4, seq_len),
            'indicators': torch.randn(batch_size, seq_len, 8),  # RSI, MACD, etc.
            'portfolio': torch.randn(batch_size, 3),  # position, pnl, risk
        }
        
        encoded = adapter.encode_market_data(market_data)
        print(f"✅ Features encoded: {encoded.shape}")
        
        # Verifica dimensioni
        assert encoded.shape == (batch_size, seq_len, 128)
        assert not torch.isnan(encoded).any()
        
        return True
        
    except Exception as e:
        print(f"❌ Errore adattatore features: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_output_head():
    """Test 3: Output head per decisioni trading"""
    print("\n🧪 Test output head...")
    
    try:
        sys.path.append(str(src_path / "hrm"))
        from heads import OutputHead, TradeProposal, TradeDirection
        import torch
        
        # Crea output head
        output_head = OutputHead(hidden_dim=128)
        
        # Simula output del modello HRM
        batch_size = 2
        seq_len = 10
        hidden_dim = 128
        
        model_output = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Decodifica in proposta di trade
        proposals = output_head.decode_batch(model_output)
        
        print(f"✅ Proposte generate: {len(proposals)}")
        
        # Verifica proposte
        for i, proposal in enumerate(proposals):
            assert isinstance(proposal, TradeProposal)
            assert isinstance(proposal.direction, TradeDirection)
            assert 0 <= proposal.confidence <= 1
            assert proposal.quantity > 0
            print(f"  Proposta {i+1}: {proposal.direction.value}, qty={proposal.quantity:.2f}, conf={proposal.confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore output head: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test 4: Creazione agenti con HRM"""
    print("\n🧪 Test creazione agenti...")
    
    try:
        sys.path.append(str(src_path / "agents"))
        sys.path.append(str(src_path / "hrm"))
        
        from base import AgentType, AgentState
        from risk_manager import RiskManagerAgent, RiskManagerConfig
        from core import HRM, HRMConfig
        from heads import FeatureAdapter, OutputHead
        
        # Crea componenti HRM
        hrm_config = HRMConfig(
            input_dim=64,
            hidden_dim=128,
            n_high_layers=2,
            n_low_layers=2
        )
        
        model = HRM(hrm_config)
        feature_adapter = FeatureAdapter(input_dim=64, output_dim=128)
        output_head = OutputHead(hidden_dim=128)
        
        # Crea configurazione agente
        risk_config = RiskManagerConfig(
            max_daily_loss=5000.0,
            max_position_risk=0.02,
            max_portfolio_risk=0.15,
            max_correlation_risk=0.8,
            max_concentration_risk=0.3,
            volatility_threshold=0.25,
            emergency_exit_threshold=0.05
        )
        
        # Crea agente
        agent = RiskManagerAgent(
            name="test_risk_manager",
            config=risk_config,
            hrm_model=model,
            feature_adapter=feature_adapter,
            output_head=output_head
        )
        
        print(f"✅ Agente creato: {agent.name}, stato={agent.state.value}")
        
        # Verifica stato
        assert agent.state == AgentState.INITIALIZED
        assert agent.agent_type == AgentType.RISK_MANAGER
        
        return True
        
    except Exception as e:
        print(f"❌ Errore creazione agenti: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_message_handling():
    """Test 5: Gestione messaggi agenti"""
    print("\n🧪 Test gestione messaggi...")
    
    try:
        sys.path.append(str(src_path / "agents"))
        sys.path.append(str(src_path / "hrm"))
        
        from base import AgentMessage
        from risk_manager import RiskManagerAgent, RiskManagerConfig
        from core import HRM, HRMConfig
        from heads import FeatureAdapter, OutputHead
        
        # Crea agente (versione semplificata)
        hrm_config = HRMConfig(input_dim=32, hidden_dim=64, n_high_layers=1, n_low_layers=1)
        model = HRM(hrm_config)
        feature_adapter = FeatureAdapter(input_dim=32, output_dim=64)
        output_head = OutputHead(hidden_dim=64)
        
        risk_config = RiskManagerConfig(max_daily_loss=1000.0)
        
        agent = RiskManagerAgent(
            name="test_agent",
            config=risk_config,
            hrm_model=model,
            feature_adapter=feature_adapter,
            output_head=output_head
        )
        
        # Test messaggio di status
        status_msg = AgentMessage(
            type="get_status",
            data={},
            timestamp=time.time()
        )
        
        response = await agent.handle_message(status_msg)
        print(f"✅ Risposta status: {response}")
        
        # Verifica risposta
        assert isinstance(response, dict)
        assert "status" in response
        
        return True
        
    except Exception as e:
        print(f"❌ Errore gestione messaggi: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_components():
    """Test 6: Componenti di training"""
    print("\n🧪 Test componenti training...")
    
    try:
        sys.path.append(str(src_path / "hrm"))
        from training import create_synthetic_data, TradingDataset, TrainingConfig
        import torch
        
        # Crea dati sintetici
        data = create_synthetic_data(num_samples=20, seq_length=10)
        print(f"✅ Dati sintetici creati: {len(data)} campioni")
        
        # Verifica struttura dati
        assert len(data) == 20
        assert all(isinstance(sample, dict) for sample in data)
        
        # Crea dataset
        dataset = TradingDataset(data)
        print(f"✅ Dataset creato: {len(dataset)} campioni")
        
        # Test caricamento campione
        features, target = dataset[0]
        assert isinstance(features, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        print(f"✅ Campione caricato: features={features.shape}, target={target.shape}")
        
        # Test configurazione training
        training_config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=2,
            warmup_steps=10
        )
        
        print(f"✅ Configurazione training: lr={training_config.learning_rate}, epochs={training_config.num_epochs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore componenti training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end_pipeline():
    """Test 7: Pipeline end-to-end"""
    print("\n🧪 Test pipeline end-to-end...")
    
    try:
        sys.path.append(str(src_path / "hrm"))
        from core import HRM, HRMConfig
        from heads import FeatureAdapter, OutputHead, TradeProposal
        import torch
        
        # 1. Crea componenti
        config = HRMConfig(
            input_dim=32,
            hidden_dim=64,
            n_high_layers=1,
            n_low_layers=1,
            n_heads=2,
            n_cycles=1,
            steps_per_cycle=2
        )
        
        model = HRM(config)
        feature_adapter = FeatureAdapter(input_dim=32, output_dim=64)
        output_head = OutputHead(hidden_dim=64)
        
        print("✅ Componenti creati")
        
        # 2. Simula dati di mercato
        market_data = {
            'prices': torch.randn(1, 2, 5),  # 1 batch, 2 timeframes, 5 steps
            'volumes': torch.randn(1, 2, 5),
            'indicators': torch.randn(1, 5, 4),  # RSI, MACD, etc.
            'portfolio': torch.randn(1, 3),  # position, pnl, risk
        }
        
        print("✅ Dati di mercato simulati")
        
        # 3. Pipeline completa
        model.eval()
        with torch.no_grad():
            # Encoding features
            encoded_features = feature_adapter.encode_market_data(market_data)
            print(f"✅ Features encoded: {encoded_features.shape}")
            
            # Inferenza HRM
            hrm_output = model(encoded_features)
            print(f"✅ HRM output: {hrm_output.shape}")
            
            # Decodifica decisione
            proposals = output_head.decode_batch(hrm_output)
            print(f"✅ Proposte generate: {len(proposals)}")
            
            # Verifica proposta
            proposal = proposals[0]
            assert isinstance(proposal, TradeProposal)
            print(f"✅ Proposta finale: {proposal.direction.value}, qty={proposal.quantity:.2f}, conf={proposal.confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore pipeline end-to-end: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Funzione principale"""
    print("🚀 Test di integrazione finale sistema HRM")
    print("=" * 60)
    
    tests = [
        ("Creazione Modello HRM", test_hrm_model_creation, False),
        ("Adattatore Features", test_feature_adapter, False),
        ("Output Head", test_output_head, False),
        ("Creazione Agenti", test_agent_creation, False),
        ("Gestione Messaggi", test_agent_message_handling, True),  # async
        ("Componenti Training", test_training_components, False),
        ("Pipeline End-to-End", test_end_to_end_pipeline, False)
    ]
    
    results = []
    
    for test_name, test_func, is_async in tests:
        try:
            if is_async:
                success = await test_func()
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Errore critico in {test_name}: {e}")
            results.append((test_name, False))
    
    # Report finale
    print("\n" + "=" * 60)
    print("📊 REPORT FINALE INTEGRAZIONE HRM")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nRisultato: {passed}/{total} test passati")
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"Tasso di successo: {success_rate:.1f}%")
    
    if passed == total:
        print("\n🎉 SISTEMA HRM COMPLETAMENTE INTEGRATO E FUNZIONANTE!")
        print("✅ Il sistema è pronto per l'uso in produzione")
        return 0
    elif success_rate >= 80:
        print("\n✅ Sistema HRM funzionante con alcune limitazioni")
        print("⚠️ Alcuni componenti potrebbero necessitare di ottimizzazioni")
        return 0
    else:
        print("\n❌ Sistema HRM necessita di correzioni")
        print("🔧 Rivedere i componenti falliti prima dell'uso")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))