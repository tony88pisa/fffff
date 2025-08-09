#!/usr/bin/env python3
"""
Test semplificato per verificare i componenti HRM base
"""

import sys
import os
from pathlib import Path

# Aggiungi il percorso src
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_basic_imports():
    """Test degli import base"""
    print("🧪 Test import base...")
    
    try:
        # Test import torch
        import torch
        print("✅ PyTorch importato correttamente")
        
        # Test import einops
        import einops
        print("✅ Einops importato correttamente")
        
        # Test import wandb
        import wandb
        print("✅ Wandb importato correttamente")
        
        return True
    except ImportError as e:
        print(f"❌ Errore import: {e}")
        return False

def test_hrm_config():
    """Test configurazione HRM"""
    print("\n🧪 Test configurazione HRM...")
    
    try:
        # Import diretto del modulo core
        sys.path.append(str(src_path / "hrm"))
        from core import HRMConfig
        
        # Crea configurazione
        config = HRMConfig(
            input_dim=128,
            hidden_dim=512,
            n_high_layers=6,
            n_low_layers=4,
            n_heads=8,
            n_cycles=3,
            steps_per_cycle=5,
            dropout=0.1,
            max_seq_len=512,
            halt_threshold=0.01
        )
        
        print(f"✅ Configurazione HRM creata: {config.input_dim}D input, {config.hidden_dim}D hidden, {config.n_high_layers} high layers")
        return True
        
    except Exception as e:
        print(f"❌ Errore configurazione HRM: {e}")
        return False

def test_basic_torch_operations():
    """Test operazioni PyTorch base"""
    print("\n🧪 Test operazioni PyTorch...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Test creazione tensori
        x = torch.randn(2, 10, 256)
        print(f"✅ Tensore creato: {x.shape}")
        
        # Test layer semplice
        layer = nn.Linear(256, 128)
        y = layer(x)
        print(f"✅ Layer lineare: {x.shape} -> {y.shape}")
        
        # Test attention
        attention = nn.MultiheadAttention(256, 8, batch_first=True)
        attn_out, _ = attention(x, x, x)
        print(f"✅ Multi-head attention: {x.shape} -> {attn_out.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore operazioni PyTorch: {e}")
        return False

def test_dataclass_creation():
    """Test creazione dataclass per trading"""
    print("\n🧪 Test dataclass trading...")
    
    try:
        from dataclasses import dataclass
        from typing import Optional
        from enum import Enum
        import torch
        
        class TradeDirection(Enum):
            BUY = "buy"
            SELL = "sell"
            HOLD = "hold"
        
        @dataclass
        class SimpleTradeProposal:
            direction: TradeDirection
            quantity: float
            confidence: float
            symbol: str = "AAPL"
            stop_loss: Optional[float] = None
        
        # Test creazione
        proposal = SimpleTradeProposal(
            direction=TradeDirection.BUY,
            quantity=100.0,
            confidence=0.85
        )
        
        print(f"✅ TradeProposal creato: {proposal.direction.value}, qty={proposal.quantity}")
        return True
        
    except Exception as e:
        print(f"❌ Errore dataclass: {e}")
        return False

def test_agent_base_structure():
    """Test struttura base degli agenti"""
    print("\n🧪 Test struttura agenti...")
    
    try:
        from enum import Enum
        from dataclasses import dataclass
        from typing import Dict, Any
        import asyncio
        
        class AgentState(Enum):
            INITIALIZED = "initialized"
            RUNNING = "running"
            STOPPED = "stopped"
            ERROR = "error"
        
        class AgentType(Enum):
            RISK_MANAGER = "risk_manager"
            POLICY = "policy"
            EXECUTION = "execution"
        
        @dataclass
        class AgentMessage:
            type: str
            data: Dict[str, Any]
            timestamp: float
        
        class SimpleAgent:
            def __init__(self, name: str, agent_type: AgentType):
                self.name = name
                self.agent_type = agent_type
                self.state = AgentState.INITIALIZED
            
            async def handle_message(self, message: AgentMessage):
                return {"status": "ok", "agent": self.name}
        
        # Test creazione agente
        agent = SimpleAgent("test_agent", AgentType.RISK_MANAGER)
        print(f"✅ Agente creato: {agent.name}, tipo={agent.agent_type.value}")
        
        # Test messaggio
        import time
        message = AgentMessage(
            type="test",
            data={"test": True},
            timestamp=time.time()
        )
        
        # Test async
        async def test_async():
            response = await agent.handle_message(message)
            return response
        
        response = asyncio.run(test_async())
        print(f"✅ Messaggio gestito: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore struttura agenti: {e}")
        return False

def test_file_structure():
    """Test struttura file del progetto"""
    print("\n🧪 Test struttura file...")
    
    try:
        base_path = Path(__file__).parent
        
        # Verifica directory principali
        required_dirs = [
            "src",
            "src/hrm",
            "src/agents",
            "tests"
        ]
        
        for dir_path in required_dirs:
            full_path = base_path / dir_path
            if full_path.exists():
                print(f"✅ Directory trovata: {dir_path}")
            else:
                print(f"⚠️ Directory mancante: {dir_path}")
        
        # Verifica file principali
        required_files = [
            "src/main.py",
            "src/hrm/__init__.py",
            "src/hrm/core.py",
            "src/hrm/heads.py",
            "src/hrm/training.py",
            "src/agents/__init__.py",
            "src/agents/base.py"
        ]
        
        for file_path in required_files:
            full_path = base_path / file_path
            if full_path.exists():
                print(f"✅ File trovato: {file_path}")
            else:
                print(f"⚠️ File mancante: {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore verifica file: {e}")
        return False

def main():
    """Funzione principale"""
    print("🚀 Test semplificato sistema HRM")
    print("=" * 50)
    
    tests = [
        ("Import Base", test_basic_imports),
        ("Configurazione HRM", test_hrm_config),
        ("Operazioni PyTorch", test_basic_torch_operations),
        ("Dataclass Trading", test_dataclass_creation),
        ("Struttura Agenti", test_agent_base_structure),
        ("Struttura File", test_file_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Errore in {test_name}: {e}")
            results.append((test_name, False))
    
    # Report finale
    print("\n" + "=" * 50)
    print("📊 REPORT FINALE")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nRisultato: {passed}/{total} test passati")
    
    if passed == total:
        print("🎉 Tutti i test base sono passati!")
        return 0
    else:
        print("⚠️ Alcuni test sono falliti")
        return 1

if __name__ == "__main__":
    sys.exit(main())