import matplotlib.pyplot as plt
import numpy as np

def init_plot_2D(fig=None, lim_from=-1, lim_to=7):
  if fig is None:
    fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(121)
  ax.set_box_aspect(1)
  ax.set_xlim([lim_from, lim_to])
  ax.set_ylim([lim_from, lim_to])
  ax.set_xlabel("$x_I$")
  ax.set_ylabel("$y_I$")

  ax.set_axisbelow(True)
  ax.grid()

  ax2 = fig.add_subplot(122)
  ax2.set_box_aspect(1)
  ax2.set_xlim([lim_from, lim_to])
  ax2.set_ylim([lim_from, lim_to])
  ax2.set_xlabel("$x_I$")
  ax2.set_ylabel("$y_I$")

  ax2.set_axisbelow(True)
  ax2.grid()

  return fig, ax, ax2

def R(theta):
  return np.array([
    [np.cos(theta), np.sin(theta), 0],
    [-np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
  ], dtype=float)

def update_wedge(wedge_patch, I_xi):
  wedge_patch.center = I_xi[:2]
  wedge_patch.theta1 = np.rad2deg(I_xi[2]) + 10
  wedge_patch.theta2 = np.rad2deg(I_xi[2]) - 10
  wedge_patch._recompute_path()

def normalize_angle(theta):
  return np.arctan2(np.sin(theta), np.cos(theta))

def log_odds_to_prob(log_odds):
  return 1.0 - (1.0 / (1.0 + np.exp(log_odds)))

def prob_to_log_odds(probs):
  return np.log(probs / (1 - probs))

def normal_pdf(x, var, mean=0):
  sqrt_inv = 1 / np.sqrt(2 * np.pi * var ** 2) 
  exp = np.exp(-0.5 * ((x - mean) ** 2) / var ** 2)
  return sqrt_inv * exp

def sample_normal_distribution(b_squared, M):
  b = np.sqrt(b_squared)
  return 1 / 2 * np.sum(np.random.uniform(-b, b, (M, 12)), axis=1)

def distance_between_points(pointA, pointB):
  return np.sqrt((pointB[0] - pointA[0])**2 + (pointB[1] - pointA[1])**2)

def bresenham(row1, column1, row2, column2):
  x0 = column1
  y0 = -row1
  x1 = column2
  y1 = -row2
  cells = []
  dx = abs(x1 - x0)
  dy = abs(y1 - y0)
  sx = 1 if x0 < x1 else -1
  sy = 1 if y0 < y1 else -1
  err = dx - dy

  while True:
    cells.append((int(-y0), int(x0)))
    if equal(x0, x1) and equal(y0, y1):
      break
    e2 = 2 * err
    if greater_than(e2, -dy):
      err -= dy
      x0 += sx
    if less_than(e2, dx):
      err += dx
      y0 += sy

  return cells

def print_map(map, start_cell=(-1, -1)):
  print()
  for i in range(map.num_of_rows):
    for j in range(map.num_of_columns):
      if (i, j) == start_cell:
        print("x ", end="")
        continue
      if greater_or_equal_than(map.log_odds_probabilities[i, j], map.l_occupied):
        print("1 ", end="")
      elif less_than(map.l_free, map.log_odds_probabilities[i, j]) and less_than(map.log_odds_probabilities[i, j], map.l_occupied):
        print("- ", end="")
      else:
        print("0 ", end="")
    print()
  print()

def less_than(value1, value2, eps = 1e-6):
  return value1 - value2 < -eps

def greater_than(value1, value2, eps = 1e-6):
  return value1 - value2 > eps

def equal(value1, value2, eps = 1e-6):
  return abs(value1 - value2) < eps

def less_or_equal_than(value1, value2):
  return less_than(value1, value2) or equal(value1, value2)

def greater_or_equal_than(value1, value2):
  return greater_than(value1, value2) or equal(value1, value2)

def obstacle_close(measurements, distance=0.7):
  for z_t_k in measurements:
    if less_or_equal_than(z_t_k, distance):
      return True
  return False
