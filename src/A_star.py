import numpy as np
from shared import greater_than, less_than, less_or_equal_than

def update_cell_f_hat_if_in_open_set(cell, open_set, new_cell_f_hat):
  for cell_in_set in open_set:
    if cell_in_set[1] == cell:
      if greater_than(cell_in_set[0], new_cell_f_hat):
        cell_in_set[0] = new_cell_f_hat
      return True
  return False

def find_cell_neighbours(cell, map):
  neighbours = []
  neighbour = (cell[0] - 1, cell[1])
  if map.inside_map(*neighbour):
    neighbours.append(neighbour)
  neighbour = (cell[0] + 1, cell[1])
  if map.inside_map(*neighbour):
    neighbours.append(neighbour)
  neighbour = (cell[0], cell[1] - 1)
  if map.inside_map(*neighbour):
    neighbours.append(neighbour)
  neighbour = (cell[0], cell[1] + 1)
  if map.inside_map(*neighbour):
    neighbours.append(neighbour)
  return neighbours

def f_hat(cell_1, cell_2, map):
  x_1, y_1 = map.indices_2_coordinates(*cell_1)
  x_2, y_2 = map.indices_2_coordinates(*cell_2)
  return np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)

def reconstruct_path(parents, start, current_cell, path):
  if current_cell != start:
    path.append(current_cell)
    reconstruct_path(parents, start, parents[current_cell[0]][current_cell[1]], path)

# we assume that there must be path from the start to the goal
def A_star(start, goal, map):
  n = map.num_of_rows
  g = np.zeros((n, n), dtype=float) - 1 # -1 representing unknown
  parents = []
  for i in range(n):
    parents.append([])
    for _ in range(n):
      parents[i].append((-1, -1))

  open_set = [[f_hat(start, goal, map), start]]
  closed_set = []

  while True:
    open_set.sort(key=lambda item: item[0])
    current_cell = open_set.pop(0)
    closed_set.append(current_cell[1])

    eligible_neighbours = find_cell_neighbours(current_cell[1], map)
    if current_cell[1] == goal or goal in eligible_neighbours:
      path = []
      reconstruct_path(parents, start, current_cell[1], path)
      path.reverse()
      return path
    
    for neighbour in eligible_neighbours:
      if less_or_equal_than(map.log_odds_probabilities[neighbour[0], neighbour[1]], map.l_free) and neighbour not in closed_set:
        if less_than(g[neighbour[0], neighbour[1]], 0.0) or greater_than(g[neighbour[0], neighbour[1]], g[current_cell[1][0], current_cell[1][1]] + map.cell_width): # g is real cost
          parents[neighbour[0]][neighbour[1]] = (current_cell[1][0], current_cell[1][1])
          g[neighbour[0], neighbour[1]] = g[current_cell[1][0], current_cell[1][1]] + map.cell_width
        neighbour_f_hat = f_hat(neighbour, goal, map) + g[neighbour[0], neighbour[1]]
        if not update_cell_f_hat_if_in_open_set(neighbour, open_set, neighbour_f_hat):        
          open_set.append([neighbour_f_hat, neighbour])