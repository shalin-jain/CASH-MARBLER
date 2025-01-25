import numpy as np

# fast_step = [.4, .45, .5, .55, .6] # ID
# slow_step = [.1, .15, .2, .25, .3] # ID
# large_torque = [14, 16, 18, 20, 22] # ID
# small_torque = [4, 6, 8, 10, 12] # ID

# fast_steps = np.random.choice(fast_step, 10)
# slow_steps = np.random.choice(slow_step, 10)
# large_torques = np.random.choice(large_torque, 10)
# small_torques = np.random.choice(small_torque, 10)

# print(fast_steps)
# print(slow_steps)
# print(large_torques)
# print(small_torques)

predator_radius = [.2, .25, .3, .35, .4] # HARD
capture_radius = [.1, .125, .15, .175, .2]  # HARD

predator_radii = np.random.choice(predator_radius, 10)
capture_radii = np.random.choice(capture_radius, 10)

print(predator_radii)
print(capture_radii)
