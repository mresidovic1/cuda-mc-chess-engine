# Google Colab - MINIMALNI Vodič (4 Koraka)

## KOPIRAJ OVA 4 CELL-A U COLAB

### ✅ CELL 1: Setup
```python
# Proveri GPU
!nvidia-smi

# Instaliraj SVE
!apt-get update -qq && apt-get install -qq cmake build-essential libomp-dev

# Provera
!cmake --version
!g++ --version
```

---

### ✅ CELL 2: Preuzmi Kod
```python
# OPCIJA A: GitHub
!git clone https://github.com/YOUR_USERNAME/cuda-mc-chess-engine.git
%cd cuda-mc-chess-engine

# OPCIJA B: Upload ZIP (ako nemaš GitHub)
# from google.colab import files
# import zipfile
# uploaded = files.upload()
# zip_name = list(uploaded.keys())[0]
# with zipfile.ZipFile(zip_name, 'r') as z: z.extractall('.')
# %cd [ime_foldera]
```

---

### ✅ CELL 3: Build (5-10 min)
```python
# Idi u tests folder
%cd /content/cuda-mc-chess-engine/tests

# Kreiraj build folder
!mkdir -p build
%cd build

# Auto-detektuj GPU
import subprocess
cap = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                     capture_output=True, text=True).stdout.strip().replace('.', '')
print(f"🎮 GPU Compute: {cap}")

# Build
!cmake .. -DCMAKE_CUDA_ARCHITECTURES="{cap}" -DBUILD_GPU_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release

!make -j$(nproc)

# Provera
!ls -la benchmark_*_gpu 2>/dev/null || echo "⚠️ GPU builds not found, checking CPU-only..."
!ls -la benchmark_* | head -10
```

---

### ✅ CELL 4: Pokreni i Skini Rezultate
```python
%cd /content/cuda-mc-chess-engine/tests/build
!mkdir -p results

# BRZI TEST (2 min)
!./benchmark_throughput_gpu \
    --output results/throughput.csv \
    --difficulty easy \
    --time 2000

# QUALITY TEST (5 min)
!./benchmark_fixed_time_gpu \
    --output results/quality.csv \
    --times 100,500,1000 \
    --suite bratko-kopec

# Skini rezultate
from google.colab import files
files.download('results/throughput.csv')
files.download('results/quality.csv')

print("✅ GOTOVO!")
```

---

## 📊 Bonus: Vizualizacija (opciono)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load
df = pd.read_csv('results/throughput.csv')

# Plot
df.boxplot(column='throughput', by='engine', figsize=(12, 6))
plt.ylabel('Throughput (ops/sec)')
plt.title('CPU vs GPU')
plt.tight_layout()
plt.savefig('results/plot.png', dpi=150)
plt.show()

files.download('results/plot.png')
```

---

## ⚠️ Ako Nešto Ne Radi

### "NVIDIA-SMI failed"
➡️ Runtime → Change runtime type → GPU

### "CMake version too old"
```python
!pip install cmake --upgrade
```

### "Out of memory"
```python
# Smanji parametre
!./benchmark_throughput_gpu --gpu-sims 1000 --time 1000
```

### Build pada
```python
# Single-threaded za bolju dijagnostiku
!make -j1
```

---

## 🎯 Šta Dalje?

**Puni benchmarks** (30+ min):
```python
!./benchmark_throughput_gpu --output results/full_throughput.csv
!./benchmark_fixed_time_gpu --output results/full_quality.csv --times 50,100,500,1000,5000
!./benchmark_matches_gpu --output results/matches.csv --games 50 --time 1000
```

**Analiza rezultata**:
```python
df = pd.read_csv('results/throughput.csv')
print(df.groupby('engine')['throughput'].describe())
```

---

## ✨ TO JE TO!

4 cell-a = kompletan benchmark. Nema dodatnih koraka.

Za detalje pogledaj `COLAB_TUTORIAL.md`.
