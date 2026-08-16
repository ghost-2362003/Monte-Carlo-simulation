from src import stress_testing
import numpy as np

path = stress_testing.simulate_market_crash("AAPL", "2026-06-16", 0.1, 0.25, 
                          0.3, 0.5 )                                         # Stress testing with log returns

path = np.asarray(path)
path = np.round(path, 5)
print(path)

print(path.shape)
print("Count:", len(path))
print("Mean:", np.mean(path))
print("Std:", np.std(path))
print("Min:", np.min(path))
print("25%:", np.percentile(path, 25))
print("50%:", np.percentile(path, 50))
print("75%:", np.percentile(path, 75))
print("Max:", np.max(path))