import numpy as np
from shared import R, normalize_angle, distance_between_points, bresenham, prob_to_log_odds, less_or_equal_than, greater_or_equal_than, greater_than, less_than
from settings import sensor_properties, map_settings
from world import unknown_world

debug = False
class Map:
  __slots__ = [
    "cell_width", 
    "cell_height", 
    "num_of_rows", 
    "num_of_columns",
    "l_free",
    "l_occupied",
    "l_0",
    "log_odds_probabilities", 
    "occupied_cells",
    "known_cells",
    "threshold_occupied",
    "threshold_free"]

  def __init__(self):
    self.cell_width = map_settings["cell_resolution"]
    self.cell_height = map_settings["cell_resolution"]
    self.num_of_rows = map_settings["num_of_rows"]
    self.num_of_columns = map_settings["num_of_columns"]
    self.l_free = map_settings["l_free"]
    self.l_occupied = map_settings["l_occupied"]
    self.l_0 = map_settings["l_0"]
    self.log_odds_probabilities = np.zeros((self.num_of_rows, self.num_of_columns), dtype=float)
    # with these thresholds it is needed two times to declare cell opposite from current state
    # if measurements are giving the opposite scores
    self.threshold_occupied = prob_to_log_odds(map_settings["threshold_occupied"])
    self.threshold_free = prob_to_log_odds(map_settings["threshold_free"])
    self.occupied_cells = []
    self.known_cells = []
    self.init_occupied_cells()

  def init_occupied_cells(self):
    for line in unknown_world[4:]:
      occupied_cells = bresenham(
        *self.coordinates_2_indices(*line[0]),
        *self.coordinates_2_indices(*line[1])
      )
      for cell in occupied_cells:
        if cell not in self.occupied_cells:
          self.known_cells.append(cell)
          self.occupied_cells.append(cell)
          self.log_odds_probabilities[cell[0], cell[1]] = self.l_occupied
    
  def inside_map(self, row, column, offset=0):
    return 0 + offset <= row < self.num_of_rows - offset and 0 + offset <= column < self.num_of_columns - offset
  
  def indices_2_coordinates(self, row, column):
    return column * self.cell_width + self.cell_width / 2, self.cell_height * self.num_of_columns - (row * self.cell_height + self.cell_height / 2)
  
  def coordinates_2_indices(self, x, y):
    return int(np.floor((self.cell_height * self.num_of_rows - y) / self.cell_height)), int(np.floor(x / self.cell_width))
  
  def distance_between_cells(self, cellA, cellB):
    positionA = self.indices_2_coordinates(*cellA)
    positionB = self.indices_2_coordinates(*cellB)
    return distance_between_points(positionA, positionB)
  
  def find_closest_occupied_cell(self, position_row, position_column):
    min_distance = float('inf')
    closest_cell = None
    for row, column in self.occupied_cells:
      distance = self.distance_between_cells((position_row, position_column), (row, column))
      if less_than(distance, min_distance):
        min_distance = distance
        closest_cell = row, column
    return closest_cell, min_distance
  
  def occupancy_grid_mapping(self, particle_I_xi, robot):
    global debug
    for sensor_index, z_t_k in enumerate(robot.z_t):
      measurement_startpoint = np.copy(particle_I_xi[:2]) + \
        R(particle_I_xi[2])[:2, :2].T @ robot.sensors[sensor_index].position
      measurement_startpoint_cell = self.coordinates_2_indices(*measurement_startpoint)
      perception_cells = []
      measurement_endpoints_cells = []
      for subbeam_orientation in robot.sensors[sensor_index].subbeam_orientations:
        measurement_endpoint = np.copy(measurement_startpoint)
        measurement_endpoint += z_t_k * np.array([
          np.cos(particle_I_xi[2] + robot.sensors[sensor_index].beam_orientation + subbeam_orientation), 
          np.sin(particle_I_xi[2] + robot.sensors[sensor_index].beam_orientation + subbeam_orientation)], 
          dtype=float
        )
        # What if measurements are outside? No problem because it will try to get every other cell
        # inside map
        measurement_endpoint_cell = self.coordinates_2_indices(*measurement_endpoint)
        perception_beam_cells = bresenham(*measurement_startpoint_cell, *measurement_endpoint_cell)
        for cell in perception_beam_cells:
          if cell not in perception_cells:
            perception_cells.append(cell)
        if measurement_endpoint_cell not in measurement_endpoints_cells:
          measurement_endpoints_cells.append(measurement_endpoint_cell)
      for cell_row, cell_column in perception_cells:
        if (cell_row, cell_column) not in self.known_cells and self.inside_map(cell_row, cell_column):
          inverse_sensor_model = self.inverse_sensor_model((cell_row, cell_column), particle_I_xi, robot)
          # if (cell_row, cell_column) in measurement_endpoints_cells and greater_than(sensor_properties["z_max"] - z_t_k, 0.05):
            # inverse_sensor_model = self.l_occupied
          if debug and distance_between_points(robot.I_xi[:2], particle_I_xi[:2]) < 0.5:
            print("Cell in particles's perception:", (cell_row, cell_column))
            print("Inverse sensor model", inverse_sensor_model)
            print("Distance from origin", self.distance_between_cells((cell_row, cell_column), self.coordinates_2_indices(*particle_I_xi[:2])))
          self.log_odds_probabilities[cell_row, cell_column] += inverse_sensor_model - self.l_0
          self.log_odds_probabilities[cell_row, cell_column] = max(self.threshold_free, min(self.log_odds_probabilities[cell_row, cell_column], self.threshold_occupied))
          if greater_or_equal_than(self.log_odds_probabilities[cell_row, cell_column], self.l_occupied) and not (cell_row, cell_column) in self.occupied_cells:
            self.occupied_cells.append((cell_row, cell_column))
          elif less_than(self.log_odds_probabilities[cell_row, cell_column], self.l_occupied) \
            and (cell_row, cell_column) in self.occupied_cells:
            self.occupied_cells.remove((cell_row, cell_column))

  def inverse_sensor_model(self, measured_cell, particle_I_xi, robot):
    measured_cell_x, measured_cell_y = self.indices_2_coordinates(*measured_cell)
    r = distance_between_points((measured_cell_x, measured_cell_y), particle_I_xi[:2])
    phi = normalize_angle(np.arctan2(measured_cell_y - particle_I_xi[1], measured_cell_x - particle_I_xi[0]) - particle_I_xi[2])
    beam_orientations = np.array([sensor.beam_orientation for sensor in robot.sensors], dtype=float)
    k = np.argmin(abs(normalize_angle(phi - beam_orientations)))
    if greater_than(r, min(sensor_properties["z_max"], robot.z_t[k] + sensor_properties["obstacle_thickness"] / 2.0)) \
      or greater_than(abs(normalize_angle(phi - robot.sensors[k].beam_orientation)), sensor_properties["view_angle"] / 2.0):
      return self.l_0
    if less_than(robot.z_t[k], sensor_properties["z_max"]) and less_than(abs(r - robot.z_t[k]), sensor_properties["obstacle_thickness"] / 2.0):
      return self.l_occupied
    if less_or_equal_than(r, robot.z_t[k]):
      return self.l_free
    # this is rare corner case when precision in decimals are making decisions
    return self.l_0
  
  def synchronize_map(self):
    for cell in self.known_cells:
      self.log_odds_probabilities[cell[0], cell[1]] = self.l_occupied
    self.occupied_cells = []
    for i in range(self.num_of_rows):
      for j in range(self.num_of_columns):
        if greater_or_equal_than(self.log_odds_probabilities[i, j], self.l_occupied):
          self.occupied_cells.append((i, j))