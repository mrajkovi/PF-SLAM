import numpy as np

sensor_properties = {
  "z_max": 1.5,
  "view_angle": np.pi / 8,
  "obstacle_thickness": 0.1,
  "sigma_hit": 0.85,
  "z_hit": 0.95,
  "z_rand": 0.01,
  "z_maximum": 0.04,
  "z_eps": 0.01
}

robot_settings = {
  "alpha1": 0.1,
  "alpha2": 0.1,
  "alpha3": 0.6,
  "alpha4": 0.1,
  "alpha5": 0.6,
  "alpha6": 0.6,
  "l": 0.065,
  "r": 0.03,
  "cpr": 300,
  "num_sensors": 5,
  "num_subbeams": 7,
  "num_of_approximators": 20
}

map_settings = {
  "cell_resolution": 0.1,
  "num_of_rows": 60,
  "num_of_columns": 60,
  "l_free": -0.25,
  "l_occupied": 0.9,
  "l_0": 0.0,
  "threshold_occupied": 0.8, # 80%
  "threshold_free": 0.2 # 20%
}