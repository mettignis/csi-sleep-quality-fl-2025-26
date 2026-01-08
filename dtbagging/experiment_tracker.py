"""
Script per gestire esperimenti multipli e tracciare risultati
"""
import json
from pathlib import Path
from datetime import datetime
import shutil

class ExperimentTracker:  
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(__file__).parent / base_dir
        self.base_dir.mkdir(exist_ok=True)
    
    def create_experiment(self, name, description, config):
        exp_dir = self.base_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        # Salva configurazione
        metadata = {
            "name": name,
            "description": description,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "kaggle_score": None,
            "kaggle_position": None
        }
        
        with open(exp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Esperimento creato: {name}")
        print(f"Directory: {exp_dir}")
        
        return exp_dir
    
    def save_results(self, exp_name, kaggle_score, kaggle_position, notes=""):
        exp_dir = self.base_dir / exp_name
        metadata_path = exp_dir / "metadata.json"
        
        if not metadata_path.exists():
            print(f"Esperimento non trovato: {exp_name}")
            return
        
        # Aggiorna metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata["kaggle_score"] = kaggle_score
        metadata["kaggle_position"] = kaggle_position
        metadata["notes"] = notes
        metadata["updated_at"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Risultati salvati per {exp_name}")
        print(f"Score: {kaggle_score}, Position: {kaggle_position}")
    
    def list_experiments(self):      
        experiments = []
        
        for exp_dir in sorted(self.base_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            metadata_path = exp_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            experiments.append(metadata)
        
        if not experiments:
            print("Nessun esperimento trovato.")
            return
        
        # Ordina per score (migliore prima)
        experiments = sorted(
            experiments,
            key=lambda x: x.get('kaggle_score', 999),
            reverse=False
        )
        
        for i, exp in enumerate(experiments, 1):
            score = exp.get('kaggle_score', 'N/A')
            pos = exp.get('kaggle_position', 'N/A')
            
            print(f"\n{i}. {exp['name']}")
            print(f"{exp['description']}")
            print(f"Score: {score}  |  Posizione: {pos}")
            
            # Mostra config principali
            config = exp.get('config', {})
            if config:
                print(f"Config: ", end="")
                key_params = []
                for k, v in config.items():
                    if k in ['algorithm', 'n_estimators', 'max_depth', 'num_rounds']:
                        key_params.append(f"{k}={v}")
                print(", ".join(key_params))
            
    def copy_files_to_experiment(self, exp_name, model_file, submission_file):
        exp_dir = self.base_dir / exp_name
        
        if model_file.exists():
            shutil.copy(model_file, exp_dir / model_file.name)
        
        if submission_file.exists():
            shutil.copy(submission_file, exp_dir / submission_file.name)

def main():
    tracker = ExperimentTracker()
    
    # Lista esperimenti esistenti
    tracker.list_experiments()

if __name__ == "__main__":
    main()
