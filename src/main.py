import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from timeit import default_timer as timer

import colors
from shared import init_plot_2D, update_wedge, normalize_angle, log_odds_to_prob, distance_between_points, less_than, print_map
from robot import Robot
from map import Map
from motion import sample_motion_model_odometry
from sampler import random_resampling
from world import known_world
from inside import inside_polygon
from particles import Particles
from planner import path_planner
from settings import robot_settings

debug = True
counter_resample = 0
def animate(i, robot, estimated_robot, enc_robot, particles, estimated_map, shapes, full_known_lines_env, dt):
  global debug, counter_resample
  # to get best navigation path use only known obstacles at start
  R_xi_dot = path_planner(np.array([enc_robot.z_t[2]], dtype=float), enc_robot.I_xi, Map())

  start = timer()
  phis_dot = robot.R_inverse_kinematics(R_xi_dot)
  I_xi_dot = robot.forward_kinematics(phis_dot)
  robot.update_state(I_xi_dot, dt)

  if debug:
    print("Original robot:", robot.I_xi)

  deltas_c = enc_robot.read_encoders(phis_dot, dt)
  phis_dot_enc = enc_robot.enc_deltas_to_phis_dot(deltas_c, dt)
  I_xi_dot_enc = enc_robot.forward_kinematics(phis_dot_enc)

  I_xi_prev = enc_robot.I_xi.copy()
  enc_robot.update_state(I_xi_dot_enc, dt)
  I_xi = enc_robot.I_xi.copy()

  if debug:
    print("Encoded robot:", enc_robot.I_xi)
  
  simulation_time = timer()
  measurement = enc_robot.simulate_measurement(full_known_lines_env)
  if debug:
    print("Measurement from encoded robot:", measurement)
  enc_robot.update_cones(measurement)
  end_simulation = timer()
  if debug:
    print("Measurement simulation lasted:", end_simulation - simulation_time)

  update_wedge(shapes[1], robot.I_xi)
  update_wedge(shapes[3], enc_robot.I_xi)

  start_sampling = timer()  
  particles.I_xis = sample_motion_model_odometry(np.vstack([I_xi_prev, I_xi]), particles.I_xis)
  end_sampling = timer()
  if debug:
    print("Sampling motion model lasted:", end_sampling - start_sampling)
  particles.normalize_angles()

  if debug:
    min_distance = float("inf")
    m = 0
    for particle_I_xi in particles.I_xis:
      if distance_between_points(enc_robot.I_xi[:2], particle_I_xi[:2]) < min_distance:
        min_distance = distance_between_points(enc_robot.I_xi[:2], particle_I_xi[:2])
      if distance_between_points(enc_robot.I_xi[:2], particle_I_xi[:2]) < 0.5:
        m += 1
    print("Min distance:", min_distance)
    print("No. of particles inside 0.5m:", m)

  particles.update_colors(full_known_lines_env)
  shapes[4].set_offsets(particles.I_xis[:, :2])
  shapes[4].set_color(particles.particle_colors)

  measurement_model_timer_start = timer()
  particles.measurement_model(enc_robot)
  particles.weights[~inside_polygon(particles.I_xis, full_known_lines_env[0:4])] = 0.0
  particles.normalize_weights()
  measurement_model_timer_end = timer()

  if debug:
    print("Measurement lasted:", measurement_model_timer_end - measurement_model_timer_start)
  
  occupancy_timer_start = timer()
  particles.occupancy_grid_mapping(enc_robot)
  occupancy_timer_end = timer()
  if debug:
    print("Occupancy grid mapping lasted:", occupancy_timer_end - occupancy_timer_start)

  # we will wait with resampling step
  if (less_than(1 / np.sum(particles.weights ** 2), particles.M / 2) and i > 2) or counter_resample == 4:
    if debug:
      print("Resampling in process")
    random_resampling(particles)
    counter_resample = 0
  else:
    counter_resample += 1

  num_of_predictors = robot_settings["num_of_approximators"]
  best_found = np.argpartition(particles.weights, -num_of_predictors)[-num_of_predictors:]
  mean = np.mean(particles.I_xis[best_found], axis=0)
  mean[2] = normalize_angle(mean[2])
  estimated_robot.I_xi = mean
  # update_wedge(shapes[2], estimated_robot.I_xi)

  estimated_map.log_odds_probabilities = np.zeros((estimated_map.num_of_rows, estimated_map.num_of_columns), dtype=float)
  for particle_map in particles.maps[best_found]:
    estimated_map.log_odds_probabilities += particle_map.log_odds_probabilities
  estimated_map.log_odds_probabilities /= num_of_predictors
  estimated_map.synchronize_map()

  if debug:
    print_map(estimated_map, estimated_map.coordinates_2_indices(*estimated_robot.I_xi[:2]))
    print("Estimated robot position:", estimated_robot.I_xi)
    print("Distance of best estimate from encoded robot:", distance_between_points(estimated_robot.I_xi[:2], enc_robot.I_xi[:2]))

  shapes[0].set_data(np.vectorize(log_odds_to_prob)(estimated_map.log_odds_probabilities))
  end_timer = timer()

  # maybe try to scatter particles around current converge to help convergence at the end
  # if i % 120 == 0 and i > 0:
  #   n = int(particles.M * 0.4)
  #   choice = np.random.choice(particles.M, n)
  #   particles.I_xis[choice, :2] = estimated_robot.I_xi[:2] + np.random.uniform(-0.3, 0.3, (n, 2))

  print("Whole loop", i, "lasted", end_timer - start)

def simulate():
  fig, known_ax, unexplored_ax = init_plot_2D()
  num_frames = 1500
  fps = 30
  dt = 1 / fps

  shapes = []
  full_known_lines_env = known_world
  
  known_ax.add_collection(
    LineCollection(full_known_lines_env, facecolors=[colors.darkgray])
  )
  robot = Robot()

  ## PARTICLES INITIALIZATION
  M = 700
  particles_I_xis = np.random.uniform(0.05, 5.95, (M, 3))
  particles_I_xis[:, 2] = normalize_angle(particles_I_xis[:, 2])
  # particles_I_xis[:, 2] = robot.I_xi[2]
  particles = Particles(particles_I_xis)
  
  particles.update_colors(full_known_lines_env)
  particles_shape = unexplored_ax.scatter(
    particles.I_xis[:, 0], particles.I_xis[:, 1], s=0.5, color=particles.particle_colors
  )

  estimated_map = Map()
  cmap = mcolors.ListedColormap(['white', 'gray', 'black'])
  bounds = [0, 0.3, 0.7, 1]  # free < 0.3, 0.3 <= unknown < 0.7, occupied > 0.7
  norm = mcolors.BoundaryNorm(bounds, cmap.N)

  map_patch = unexplored_ax.imshow(
    np.vectorize(log_odds_to_prob)(estimated_map.log_odds_probabilities),
    cmap=cmap, 
    norm=norm, 
    origin='upper',
    extent=[0, estimated_map.num_of_columns * estimated_map.cell_width, 0, estimated_map.num_of_rows * estimated_map.cell_height]
  )
  # shapes[0] is estimated map
  shapes.append(map_patch)

  mean = np.mean(particles.I_xis, axis=0)
  mean[2] = normalize_angle(mean[2])

  estimated_robot = Robot(mean)
  enc_robot = Robot(robot.I_xi.copy())

  robot_patch_known_world = known_ax.add_patch(
    Wedge(
      robot.I_xi[:2],
      0.2,
      np.rad2deg(robot.I_xi[2]) + 10,
      np.rad2deg(robot.I_xi[2]) - 10,
      zorder=2
    )
  )

  estimated_robot_patch_known_world = None

  enc_robot_patch_known_world = known_ax.add_patch(
    Wedge(
      enc_robot.I_xi[:2],
      0.2,
      np.rad2deg(enc_robot.I_xi[2]) + 10,
      np.rad2deg(enc_robot.I_xi[2]) - 10,
      facecolor=colors.darkgray, alpha=0.7, zorder=3
    )
  )

  # shapes[1] is robot without noise in known world
  shapes.append(robot_patch_known_world)
  # shapes[2] is estimated robot from particles
  shapes.append(estimated_robot_patch_known_world)
  # shapes[3] is encoded robot from odometry in known world
  shapes.append(enc_robot_patch_known_world)
  # shapes[4] are particles
  shapes.append(particles_shape)

  sensor_measurements = robot.simulate_measurement(full_known_lines_env)
  # robot.draw_cones(known_ax, sensor_measurements)
  enc_robot.draw_cones(known_ax, sensor_measurements, colors.magenta)
  
  ani = FuncAnimation(
    fig,
    animate,
    fargs=(robot, estimated_robot, enc_robot, particles, estimated_map, shapes, full_known_lines_env, dt),
    frames=num_frames,
    interval=dt * 1000,
    repeat=False,
    blit=False,
    init_func=lambda: None
  )
  
  ani.save('C:\\Users\\MainMPC\\Documents\Fakultet\\simulation.gif', writer='imagemagick', fps=fps)

if __name__ == "__main__":
  np.random.seed(110)
  simulate()