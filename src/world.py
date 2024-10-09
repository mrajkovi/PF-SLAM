import numpy as np
from shared import R

unknown_world = np.array([
  # world
  [[0, 0], [6, 0]],
  [[6, 0], [6, 6]],
  [[6, 6], [0, 6]],
  [[0, 6], [0, 0]],
  # rectangle 1
  # [[1.5, 1.75], [4.25, 1.75]],
  [[4.25, 1.75], [4.25, 2.15]],
  [[4.25, 2.15], [1.5, 2.15]],
  [[1.5, 2.15], [1.5, 1.75]],
  # rectangle 2
  (np.array([[1, 4], [2.5, 4]], dtype=float) - np.array([1.75, 4.125], dtype=float)) @ R(-np.pi/3)[:2, :2].T + np.array([1.75, 4.125], dtype=float),
  # (np.array([[2.5, 4], [2.5, 4.25]], dtype=float) - np.array([1.75, 4.125], dtype=float)) @ R(-np.pi/3)[:2, :2].T + np.array([1.75, 4.125], dtype=float),
  (np.array([[2.5, 4.25], [1, 4.25]], dtype=float) - np.array([1.75, 4.125], dtype=float)) @ R(-np.pi/3)[:2, :2].T + np.array([1.75, 4.125], dtype=float),
  (np.array([[1, 4.25], [1, 4]], dtype=float) - np.array([1.75, 4.125], dtype=float)) @ R(-np.pi/3)[:2, :2].T + np.array([1.75, 4.125], dtype=float),  
  # rectangle 3
  # (np.array([[3.5, 4], [5, 4]], dtype=float) - np.array([4.25, 4.125], dtype=float)) @ R(np.pi/3)[:2, :2].T + np.array([4.25, 4.125], dtype=float),
  # (np.array([[5, 4], [5, 4.25]], dtype=float) - np.array([4.25, 4.125], dtype=float)) @ R(np.pi/3)[:2, :2].T + np.array([4.25, 4.125], dtype=float),
  (np.array([[5, 4.25], [3.5, 4.25]], dtype=float) - np.array([4.25, 4.125], dtype=float)) @ R(np.pi/3)[:2, :2].T + np.array([4.25, 4.125], dtype=float),
  # (np.array([[3.5, 4.25], [3.5, 4]], dtype=float) - np.array([4.25, 4.125], dtype=float)) @ R(np.pi/3)[:2, :2].T + np.array([4.25, 4.125], dtype=float),
  # obstacle 1
  [[2, 0.15], [3.75, 0.15]],
], dtype=float)

known_world = np.array([
  # world
  [[0, 0], [6, 0]],
  [[6, 0], [6, 6]],
  [[6, 6], [0, 6]],
  [[0, 6], [0, 0]],
  # rectangle 1
  [[1.5, 1.75], [4.25, 1.75]],
  [[4.25, 1.75], [4.25, 2.15]],
  [[4.25, 2.15], [1.5, 2.15]],
  [[1.5, 2.15], [1.5, 1.75]],
  # rectangle 2
  (np.array([[1, 4], [2.5, 4]], dtype=float) - np.array([1.75, 4.125], dtype=float)) @ R(-np.pi/3)[:2, :2].T + np.array([1.75, 4.125], dtype=float),
  (np.array([[2.5, 4], [2.5, 4.25]], dtype=float) - np.array([1.75, 4.125], dtype=float)) @ R(-np.pi/3)[:2, :2].T + np.array([1.75, 4.125], dtype=float),
  (np.array([[2.5, 4.25], [1, 4.25]], dtype=float) - np.array([1.75, 4.125], dtype=float)) @ R(-np.pi/3)[:2, :2].T + np.array([1.75, 4.125], dtype=float),
  (np.array([[1, 4.25], [1, 4]], dtype=float) - np.array([1.75, 4.125], dtype=float)) @ R(-np.pi/3)[:2, :2].T + np.array([1.75, 4.125], dtype=float),  
  # rectangle 3
  (np.array([[3.5, 4], [5, 4]], dtype=float) - np.array([4.25, 4.125], dtype=float)) @ R(np.pi/3)[:2, :2].T + np.array([4.25, 4.125], dtype=float),
  (np.array([[5, 4], [5, 4.25]], dtype=float) - np.array([4.25, 4.125], dtype=float)) @ R(np.pi/3)[:2, :2].T + np.array([4.25, 4.125], dtype=float),
  (np.array([[5, 4.25], [3.5, 4.25]], dtype=float) - np.array([4.25, 4.125], dtype=float)) @ R(np.pi/3)[:2, :2].T + np.array([4.25, 4.125], dtype=float),
  (np.array([[3.5, 4.25], [3.5, 4]], dtype=float) - np.array([4.25, 4.125], dtype=float)) @ R(np.pi/3)[:2, :2].T + np.array([4.25, 4.125], dtype=float),
  # obstacle 1
  [[2, 0.15], [3.75, 0.15]],
], dtype=float)

known_world4 = np.array([
  # world
  [[0, 0], [6, 0]],
  [[6, 0], [6, 6]],
  [[6, 6], [0, 6]],
  [[0, 6], [0, 0]],
  # rectangle 1
  [[1.5, 1.75], [4.25, 1.75]],
  [[4.25, 1.75], [4.25, 2.15]],
  [[4.25, 2.15], [1.5, 2.15]],
  [[1.5, 2.15], [1.5, 1.75]],
  # rectangle 2
  [[1, 4], [1.25, 4]],
  [[1.25, 4], [1.25, 5]],
  [[1.25, 5], [1, 5]],
  [[1, 5], [1, 4]],
  # rectangle 3
  [[3.5, 4], [3.75, 4]],
  [[3.75, 4], [3.75, 5]],
  [[3.75, 5], [3.5, 5]],
  [[3.5, 5], [3.5, 4]],
  # obstacle 1
  [[2, 0.15], [3.75, 0.15]],
], dtype=float)


unknown_world4 = np.array([
  # world
  [[0, 0], [6, 0]],
  [[6, 0], [6, 6]],
  [[6, 6], [0, 6]],
  [[0, 6], [0, 0]],
  # rectangle 1
  [[1.5, 1.75], [4.25, 1.75]],
  [[4.25, 1.75], [4.25, 2.15]],
  [[4.25, 2.15], [1.5, 2.15]],
  # [[1.5, 2.15], [1.5, 1.75]],
  # rectangle 2
  # [[1, 4], [1.25, 4]],
  [[1.25, 4], [1.25, 5]],
  # [[1.25, 5], [1, 5]],
  [[1, 5], [1, 4]],
  # rectangle 3
  [[3.5, 4], [3.75, 4]],
  # [[3.75, 4], [3.75, 5]],
  [[3.75, 5], [3.5, 5]],
  [[3.5, 5], [3.5, 4]],
  # obstacle 1
  [[2, 0.15], [3.75, 0.15]],
], dtype=float)