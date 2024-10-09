import numpy as np
from map import Map
from shared import R, normal_pdf, distance_between_points, normalize_angle, greater_than
from inside import inside_polygon 
from settings import sensor_properties
import colors
from copy import deepcopy

debug = False

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
    # inside_obstacle = inside_polygon(self.I_xis, lines_env[4:8]) \
    #   | inside_polygon(self.I_xis, lines_env[8:12]) \
    #   | inside_polygon(self.I_xis, lines_env[12:16])

    outside = ~inside_polygon(self.I_xis, lines_env[0:4])
    # self.particle_colors[outside | inside_obstacle] = colors.red
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
    global debug
    for index, map in enumerate(self.maps):
      p = 1
      for sensor_index, z_t_k in enumerate(robot.z_t):
        if greater_than(abs(z_t_k - sensor_properties["z_max"]), 0.05):
          # subbeams will not be included because it could give high score to particles
          # that are misplaced
          measurement_endpoints = np.copy(self.I_xis[index][:2])
          measurement_endpoints += \
            R(self.I_xis[index][2])[:2, :2].T @ robot.sensors[sensor_index].position + \
            z_t_k * np.array([
                np.cos(self.I_xis[index][2] + robot.sensors[sensor_index].beam_orientation),
                np.sin(self.I_xis[index][2] + robot.sensors[sensor_index].beam_orientation)], 
                dtype=float
            )
          position_row, position_column = map.coordinates_2_indices(*measurement_endpoints)
          # it could happen that sensors returns poor value so we should take into account that for each particle
          if not map.inside_map(position_row, position_column):
            p *= 0.4
            continue
          if len(map.occupied_cells) == 0:
            continue
          closest_cell, distance = map.find_closest_occupied_cell(position_row, position_column)
          distance_prob = normal_pdf(distance, sensor_properties["sigma_hit"])
          p *= (sensor_properties["z_hit"] * distance_prob + sensor_properties["z_rand"] / sensor_properties["z_maximum"])
          if debug and (distance_between_points(robot.I_xi[:2], self.I_xis[index][:2]) < 0.5 or p > 1):
            print("Particle:", self.I_xis[index], map.coordinates_2_indices(*self.I_xis[index][:2]))
            print("Closest cell is ", closest_cell, "from", (position_row, position_column))
            print("Distance, distance_prob, total_prob:", distance, distance_prob, sensor_properties["z_hit"] * distance_prob + sensor_properties["z_rand"] / sensor_properties["z_maximum"])
            
      self.weights[index] *= p
      if debug and (distance_between_points(robot.I_xi[:2], self.I_xis[index][:2]) < 0.5 or p > 1):
        print("Weight:", self.weights[index])
      
    if debug:
      print("Best weight:", self.I_xis[np.argmax(self.weights)], self.weights[np.argmax(self.weights)])