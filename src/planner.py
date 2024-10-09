import numpy as np
from shared import normalize_angle, obstacle_close
from A_star import A_star
from map import Map

def prepare_map(estimated_map):
  new_map = Map()
  new_map.log_odds_probabilities[:, :] = new_map.l_free
  directions = []
  # we need to enlarge obstacles so that robot doesn't travel close to them
  for i in range(-3, 4):
    for j in range(-3, 4):
      directions.append((i, j))
  for occupied_cell_row, occupied_cell_column in estimated_map.known_cells:
    for direction in directions:
      occupied_cell_enlarged = occupied_cell_row + direction[0], occupied_cell_column + direction[1]
      if new_map.inside_map(*occupied_cell_enlarged):
        new_map.log_odds_probabilities[occupied_cell_enlarged[0], occupied_cell_enlarged[1]] = new_map.l_occupied
  directions = []
  for i in range(-2, 3):
    for j in range(-2, 3):
      directions.append((i, j))
  for occupied_cell_row, occupied_cell_column in estimated_map.occupied_cells:
    if (occupied_cell_row, occupied_cell_column) not in estimated_map.known_cells:
      for direction in directions:
        occupied_cell_enlarged = occupied_cell_row + direction[0], occupied_cell_column + direction[1]
        if new_map.inside_map(*occupied_cell_enlarged):
          new_map.log_odds_probabilities[occupied_cell_enlarged[0], occupied_cell_enlarged[1]] = new_map.l_occupied
  # we don't want to travel alongside edges so we will pretend they are obstacle
  for row in range(new_map.num_of_rows):
    for column in range(new_map.num_of_columns):
      for direction in directions:
        occupied_cell_enlarged = 0 + direction[0], column + direction[1]
        if new_map.inside_map(*occupied_cell_enlarged):
          new_map.log_odds_probabilities[occupied_cell_enlarged[0], occupied_cell_enlarged[1]] = new_map.l_occupied
        occupied_cell_enlarged = new_map.num_of_rows - 1 + direction[0], column + direction[1]
        if new_map.inside_map(*occupied_cell_enlarged):
          new_map.log_odds_probabilities[occupied_cell_enlarged[0], occupied_cell_enlarged[1]] = new_map.l_occupied
        occupied_cell_enlarged = row + direction[0], 0 + direction[1]
        if new_map.inside_map(*occupied_cell_enlarged):
          new_map.log_odds_probabilities[occupied_cell_enlarged[0], occupied_cell_enlarged[1]] = new_map.l_occupied
        occupied_cell_enlarged = row + direction[0], new_map.num_of_columns - 1 + direction[1]
        if new_map.inside_map(*occupied_cell_enlarged):
          new_map.log_odds_probabilities[occupied_cell_enlarged[0], occupied_cell_enlarged[1]] = new_map.l_occupied
  return new_map

target_cells = [(30, 50), (49, 52), (50, 30), (51, 15), (12, 10), (8, 50), (30, 48), (30, 30)]
going_forward = True
inputs = []
def path_planner(z_t, estimated_state, estimated_map):
  global target_cells, going_forward, inputs
  if len(target_cells) == 0 and len(inputs) == 0:
    # we have traveled through all points that were set
    exit()
  if going_forward and obstacle_close(z_t, distance=0.75):
    # obstacle is close so we need to recalculate path
    going_forward = False
  elif going_forward:
    # just keep going forward initially
    return np.array([0.7, 0, 0], dtype=float)

  if not obstacle_close(z_t, distance=0.15) and len(inputs) != 0:
    return inputs.pop(0)
  elif obstacle_close(z_t, distance=0.15):
    # we need to recalculate path because the current path has obstacle very close to it
    target_cell = target_cells[0]
  else:
    target_cell = target_cells.pop(0)
  
  start_cell = estimated_map.coordinates_2_indices(*estimated_state[:2])
  map_for_navigation = prepare_map(estimated_map)
  path = A_star(start_cell, target_cell, map_for_navigation)
  
  estimated_orientation = estimated_state[2]
  target_cell_visited = False
  pairs = []
  k = 3
  previous_cell = start_cell
  for index in range(k, len(path), k):
    pairs.append((previous_cell, path[index]))
    previous_cell = path[index]
    if path[index] == target_cell:
      target_cell_visited = True
  if not target_cell_visited:
    pairs.append((previous_cell, target_cell))
  for cellA, cellB in pairs:
    cellA_x, cellA_y = estimated_map.indices_2_coordinates(*cellA)
    cellB_x, cellB_y = estimated_map.indices_2_coordinates(*cellB)
    delta_theta = normalize_angle(np.arctan2(cellB_y - cellA_y, cellB_x - cellA_x) - estimated_orientation)
    estimated_orientation = normalize_angle(estimated_orientation + delta_theta)
    # we want to travel through k cells in 0.5s, 1s = 30 frames
    for _ in range(15):
      # do not normalize angle
      inputs.append(np.array([k * 0.1 * 2, 0, delta_theta * 2], dtype=float))
  return inputs.pop(0)
