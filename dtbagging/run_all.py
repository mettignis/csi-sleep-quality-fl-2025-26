"""Script per avviare simulazione Federated Learning"""
import subprocess
import sys

def main():
    print("🚀 Avvio Federated Learning con 5 client...")
    
    try:
        # Esegui la simulazione Flower
        result = subprocess.run(
            ["flwr", "run", ". "],
            check=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("⚠️ Warnings:", result.stderr)
            
        print("✅ Training completato!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante il training: {e}")
        print(f"Output: {e.stdout}")
        print(f"Errori: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()