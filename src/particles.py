import numpy as np
from map import Map
from shared import R, normal_pdf, distance_between_points, normalize_angle, greater_than
from inside import inside_polygon 
from settings import sensor_properties
import colors
from copy import deepcopy

class Particles:
  __slots__ = ["I_xis", "maps", "weights", "M", "particle_colors"]
    
  def __init__(self, I_xis):
    self.M = I_xis.shape[0]
    self.I_xis = I_xis
    self.maps = np.array([Map() for _ in range(self.M)])
    self.weights = np.ones(self.M, dtype=float) / self.M
    self.particle_colors = np.empty(self.M, dtype=object)
    
  def normalize_angles(self):
    self.I_xis[:, 2] = normalize_angle(self.I_xis[:, 2])

  def update_colors(self, lines_env):
    self.particle_colors[:] = colors.green
    outside = ~inside_polygon(self.I_xis, lines_env[0:4])
    self.particle_colors[outside] = colors.red
  
  def normalize_weights(self):
    self.weights = self.weights / np.sum(self.weights)

  def occupancy_grid_mapping(self, robot):
    for index, map in enumerate(self.maps):
      map.occupancy_grid_mapping(self.I_xis[index], robot)

  def filter_particles(self, indices):
    self.I_xis = self.I_xis[indices]
    self.weights = self.weights[indices]
    new_maps = []
    for index in indices:
      new_maps.append(deepcopy(self.maps[index]))
    self.maps = np.array(new_maps)

  def measurement_model(self, robot):
    for index, map in enumerate(self.maps):
      p = 1
      for sensor_index, z_t_k in enumerate(robot.z_t):
        if greater_than(abs(z_t_k - sensor_properties["z_max"]), 0.05):
          measurement_endpoints = np.copy(self.I_xis[index][:2])
          measurement_endpoints += \
            R(self.I_xis[index][2])[:2, :2].T @ robot.sensors[sensor_index].position + \
            z_t_k * np.array([
                np.cos(self.I_xis[index][2] + robot.sensors[sensor_index].beam_orientation),
                np.sin(self.I_xis[index][2] + robot.sensors[sensor_index].beam_orientation)], 
                dtype=float
            )
          position_row, position_column = map.coordinates_2_indices(*measurement_endpoints)
          if not map.inside_map(position_row, position_column):
            p *= 0.4
            continue
          if len(map.occupied_cells) == 0:
            continue
          _, distance = map.find_closest_occupied_cell(position_row, position_column)
          distance_prob = normal_pdf(distance, sensor_properties["sigma_hit"])
          p *= (sensor_properties["z_hit"] * distance_prob + sensor_properties["z_rand"] / sensor_properties["z_maximum"])
      self.weights[index] *= p