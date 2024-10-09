import numpy as np
from shared import R, normalize_angle
from ray_casting import ray_casting
from sensors import Sensor
import colors
from settings import sensor_properties, robot_settings

def M(l):
  return np.array([
    [1, 0, -l],
    [1, 0, l],
    [0, 1, 0]
  ], dtype=float)


def M_inv(l):
  return np.array([
    [1 / 2, 1 / 2, 0],
    [0, 0, 1],
    [-1 / 2 / l, 1 / 2 / l, 0]
  ], dtype=float)


class Robot:
  __slots__ = ["I_xi", "l", "r", "cpr", "beam_orientations", "sensors", "num_sensors", "z_t"]
    
  def __init__(self, I_xi = np.array([2.95, 4.85, -np.pi / 2], dtype=float)):
    self.I_xi = I_xi
    self.l = robot_settings["l"]
    self.r = robot_settings["r"]
    self.cpr = robot_settings["cpr"]
    self.num_sensors = robot_settings["num_sensors"]
    self.beam_orientations = np.flip(np.linspace(-np.pi/2, np.pi/2, self.num_sensors, dtype=float))
    # first sensor is left one
    self.sensors = np.array([
      Sensor(np.array([0.0, 0.0], dtype=float), beam_orientation, robot_settings["num_subbeams"]) \
        for beam_orientation in self.beam_orientations])
    self.z_t = np.ones(self.num_sensors)
        
  def I_sensor_full(self, sensor_index):
    sensor = self.sensors[sensor_index]
    beams = np.zeros((sensor.num_subbeams, 2, 2), dtype=float)
    beams[:, :, :] = self.I_xi[:2] + R(self.I_xi[2])[:2, :2].T @ sensor.position
    beams[:, 1, :] += sensor_properties["z_max"] * np.array([
      np.cos(self.I_xi[2] + sensor.beam_orientation + sensor.subbeam_orientations), 
      np.sin(self.I_xi[2] + sensor.beam_orientation + sensor.subbeam_orientations)], 
      dtype=float
    ).T

    return beams

  def simulate_measurement(self, lines_env):
    for sensor_index, sensor in enumerate(self.sensors):
      measured = ray_casting(lines_env, self.I_sensor_full(sensor_index))
      measured_res = measured[:, 1, :] - measured[:, 0, :]
      sensor.distance = min(np.sqrt(measured_res[:, 0] ** 2 + measured_res[:, 1] ** 2))
    return self.update_measurements()
        
  def update_state(self, I_xi_dot, dt):
    self.I_xi += I_xi_dot * dt
    self.I_xi[2] = normalize_angle(self.I_xi[2])
        
  def update_state_R(self, R_xi_dot, dt):
    I_xi_dot = R(self.I_xi[2]).T @ R_xi_dot
    self.update_state(I_xi_dot, dt)

  def R_inverse_kinematics(self, R_xi_dot):
    return 1 / self.r * M(self.l) @ R_xi_dot

  def forward_kinematics(self, phis_dot):
    return self.r * R(self.I_xi[2]).T @ M_inv(self.l) @ phis_dot
  
  def read_encoders(self, phis_dot, dt):
    delta_c_left = int(self.cpr * phis_dot[0] * dt / 2 / np.pi)
    delta_c_right = int(self.cpr * phis_dot[1] * dt / 2 / np.pi)
    return np.array([delta_c_left, delta_c_right], dtype=float)

  def enc_deltas_to_phis_dot(self, deltas_c, dt):
    phi_dot_l_enc = deltas_c[0] * 2 * np.pi / self.cpr / dt
    phi_dot_r_enc = deltas_c[1] * 2 * np.pi / self.cpr / dt
    return np.array([phi_dot_l_enc, phi_dot_r_enc, 0], dtype=float)
  
  def update_measurements(self):
    distances = np.array([sensor.distance for sensor in self.sensors], dtype=float)
    n = distances.shape[0]
    z_max = sensor_properties["z_max"]
    sigma_hit = sensor_properties["sigma_hit"]
    z_eps = sensor_properties["z_eps"]
    z_maximum = sensor_properties["z_maximum"]
    z_hit = sensor_properties["z_hit"]
    z_rand = sensor_properties["z_rand"]
    
    case_1 = (
      distances + np.random.normal(0, sigma_hit, n)
    ).clip(min=0, max=z_max)
    case_2 = np.random.uniform(
      z_max - z_eps, z_max + z_eps, n
    ).clip(min=0, max=z_max)
    case_3 = np.random.uniform(
      0, z_max, n
    )

    case_indices = np.random.choice(
      np.arange(3),
      (1, n),
      p = [z_hit, z_maximum, z_rand]
    ).T.repeat(3, axis=1)

    case_range = np.arange(0, 3).reshape(1, 3).repeat(
      n, axis=0
    )
    # case probabilities
    cp = (case_indices == case_range)
    # each z_t is one of the cases
    self.z_t = case_1 * cp[:, 0] + case_2 * cp[:, 1] + case_3 * cp[:, 2]
    
    # assume perfect sensors
    self.z_t = distances
    return self.z_t

  def draw_cones(self, ax, measurements, color = colors.purple):
    for sensor_index, measurement in enumerate(measurements):
      self.sensors[sensor_index].draw_cone(self.I_xi, measurement, ax, color)
  
  def update_cones(self, measurements):
    for sensor_index, measurement in enumerate(measurements):
      self.sensors[sensor_index].update_cone(self.I_xi, measurement)
    