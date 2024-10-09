import numpy as np
from shared import sample_normal_distribution

def low_variance_resampling(particles):
  M = particles.M
  indices = []

  v = sample_normal_distribution(1, 1)
  r =  v / M
  c = particles.weights[0]

  i = 0
  for m in range(M):
    # M - 1
    U = r + m * (1.0 / (M - 1))
    while (U > c):
      i += 1
      if i > M - 1:
        i = M - 1
        break
      c += particles.weights[i]
    indices.append(i)

  particles.filter_particles(indices)

def random_resampling(particles):
  indices = np.random.choice(
    np.arange(particles.M),
    particles.M, 
    p=particles.weights)
  particles.filter_particles(indices)
  particles.normalize_weights()