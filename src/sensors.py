import numpy as np
from matplotlib.patches import Wedge
from shared import R
import colors
from settings import sensor_properties

class Sensor:
  __slots__ = [
    "beam_orientation", 
    "sensors_positions", 
    "sensors_orientations", 
    "distance", 
    "view_angle",
    "position",
    "num_subbeams",
    "subbeam_orientations",
    "cone"
  ]
  
  def __init__(self, position, beam_orientation, num_subbeams):
    # orientation of the middle beam of the sensor's cone relative to the robot's heading direction
    self.beam_orientation = beam_orientation
    self.num_subbeams = num_subbeams
    self.view_angle = sensor_properties["view_angle"]
    self.position = position
    self.subbeam_orientations = \
      np.flip(np.linspace(-self.view_angle / 2, +self.view_angle / 2, self.num_subbeams, dtype=float))
    self.distance = sensor_properties["z_max"]
    self.cone = None

  def update_cone(self, I_xi_robot, z_t_k):
    self.cone.center = I_xi_robot[:2] + R(I_xi_robot[2])[:2, :2].T @ self.position
    self.cone.theta1 = np.rad2deg(I_xi_robot[2] + self.beam_orientation - self.view_angle / 2)
    self.cone.theta2 = np.rad2deg(I_xi_robot[2] + self.beam_orientation + self.view_angle / 2)
    self.cone.r = z_t_k
    self.cone._recompute_path()
  
  def draw_cone(self, I_xi_robot, z_t_k, ax, color = colors.purple):
    self.cone = ax.add_patch(
      Wedge(
        I_xi_robot[:2] + R(I_xi_robot[2])[:2, :2].T @ self.position,
        z_t_k,
        np.rad2deg(I_xi_robot[2] + self.beam_orientation - self.view_angle / 2),
        np.rad2deg(I_xi_robot[2] + self.beam_orientation + self.view_angle / 2),
        facecolor=color, alpha=0.7
      )
    )