"""Script per eseguire la simulazione di Federated Learning."""
import subprocess
import sys
import os

def run_simulation():
    """Esegue la simulazione Flower."""
    print("🚀 Avvio simulazione Federated Learning...")
    
    # Usa il comando flwr dal virtual environment
    venv_bin = os.path.join(os.path.dirname(__file__), "..", "flvenv", "bin")
    flwr_path = os.path.join(venv_bin, "flwr")
    
    # Aggiungi la directory bin del venv al PATH
    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    
    try:
        subprocess.run([
            flwr_path, "run", ".",
            "--run-config", 'train-method="bagging" num-server-rounds=10'
        ], check=True, env=env)
        
        print("\n✅ Simulazione completata!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Errore durante la simulazione: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_simulation()