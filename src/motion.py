import numpy as np
from shared import normalize_angle, sample_normal_distribution
from settings import robot_settings

def sample_motion_model_velocity(u_current, x_prev, dt):
  M = x_prev.shape[0]
  v_hat = u_current[0] + sample_normal_distribution(robot_settings["alpha1"] * abs(u_current[0]) ** 2 + robot_settings["alpha2"] * abs(u_current[1]) ** 2, M)
  omega_hat = u_current[1] + sample_normal_distribution(robot_settings["alpha3"] * abs(u_current[0]) ** 2 + robot_settings["alpha4"] * abs(u_current[1]) ** 2, M)
  gamma_hat = sample_normal_distribution(robot_settings["alpha5"] * abs(u_current[0]) ** 2 + robot_settings["alpha6"] * abs(u_current[1]) ** 2, M)

  x = x_prev[:, 0]
  y = x_prev[:, 1]
  theta = x_prev[:, 2]

  u = v_hat / omega_hat
  x_dash = x - u * np.sin(theta) + u * np.sin(theta + omega_hat * dt)
  y_dash = y + u * np.cos(theta) - u * np.cos(theta + omega_hat * dt)
  theta_dash = theta + omega_hat * dt + gamma_hat * dt

  x_current = np.vstack((x_dash, y_dash, theta_dash)).T  
  return x_current
  

def sample_motion_model_odometry(u_current, x_prev):
  M = x_prev.shape[0]

  delta_rot_1 = np.arctan2(
    u_current[1, 1] - u_current[0, 1],
    u_current[1, 0] - u_current[0, 0],
  ) - u_current[0, 2]

  delta_trans = np.sqrt(
    (u_current[0, 0] - u_current[1, 0]) ** 2 + \
    (u_current[0, 1] - u_current[1, 1]) ** 2
  )

  delta_rot_2 = u_current[1, 2] - u_current[0, 2] - delta_rot_1

  delta_rot_1 = normalize_angle(delta_rot_1)
  delta_rot_2 = normalize_angle(delta_rot_2)

  delta_hat_rot_1 = delta_rot_1 - sample_normal_distribution(
    robot_settings["alpha1"] * delta_rot_1 ** 2 + robot_settings["alpha2"] * delta_trans ** 2,
    M)
  delta_hat_trans = delta_trans - sample_normal_distribution(
    robot_settings["alpha3"] * delta_trans ** 2 + robot_settings["alpha4"] * delta_rot_1 ** 2 + robot_settings["alpha4"] * delta_rot_2 ** 2, 
    M)
  delta_hat_rot_2 = delta_rot_2 - sample_normal_distribution(
    robot_settings["alpha1"] * delta_rot_2 ** 2 + robot_settings["alpha2"] * delta_trans ** 2,
    M)

  x = x_prev[:, 0]
  y = x_prev[:, 1]
  theta = x_prev[:, 2]

  x_dash = x + delta_hat_trans * np.cos(theta + delta_hat_rot_1)
  y_dash = y + delta_hat_trans * np.sin(theta + delta_hat_rot_1)
  theta_dash = theta + delta_hat_rot_1 + delta_hat_rot_2

  x_current = np.vstack((x_dash, y_dash, theta_dash)).T
  return x_current