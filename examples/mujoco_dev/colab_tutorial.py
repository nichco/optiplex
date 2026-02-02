# import mujoco
# import time
# import itertools
# import numpy as np
# import mediapy as media
# import matplotlib.pyplot as plt

# # More legible printing from numpy.
# np.set_printoptions(precision=3, suppress=True, linewidth=100)

# xml = """
# <mujoco>
#   <worldbody>
#     <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
#     <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#   </worldbody>
# </mujoco>
# """
# model = mujoco.MjModel.from_xml_string(xml)

# print(model.ngeom)

# print(model.geom_rgba)

# try:
#   model.geom()
# except KeyError as e:
#   print(e)


# print(model.geom('green_sphere'))

# print(model.geom('green_sphere').rgba)

# id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'green_sphere')
# print(model.geom_rgba[id, :])

# print('id of "green_sphere": ', model.geom('green_sphere').id)
# print('name of geom 1: ', model.geom(1).name)
# print('name of body 0: ', model.body(0).name)

# print([model.geom(i).name for i in range(model.ngeom)])

# data = mujoco.MjData(model)

# print(data.geom_xpos)

# mujoco.mj_kinematics(model, data)
# print('raw access:\n', data.geom_xpos)

# # MjData also supports named access:
# print('\nnamed access:\n', data.geom('green_sphere').xpos)



# xml = """
# <mujoco>
#   <worldbody>
#     <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
#     <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#   </worldbody>
# </mujoco>
# """
# # Make model and data
# model = mujoco.MjModel.from_xml_string(xml)
# data = mujoco.MjData(model)

# # Make renderer, render and show the pixels
# with mujoco.Renderer(model) as renderer:
#   media.show_image(renderer.render())


# with mujoco.Renderer(model) as renderer:
#   mujoco.mj_forward(model, data)
#   renderer.update_scene(data)

#   media.show_image(renderer.render())




# xml = """
# <mujoco>
#   <worldbody>
#     <light name="top" pos="0 0 1"/>
#     <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
#     <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#   </worldbody>
# </mujoco>
# """
# model = mujoco.MjModel.from_xml_string(xml)
# data = mujoco.MjData(model)

# with mujoco.Renderer(model) as renderer:
#   mujoco.mj_forward(model, data)
#   renderer.update_scene(data)

#   media.show_image(renderer.render())





# duration = 3.8  # (seconds)
# framerate = 60  # (Hz)

# # Simulate and display video.
# frames = []
# mujoco.mj_resetData(model, data)  # Reset state and time.
# with mujoco.Renderer(model) as renderer:
#   while data.time < duration:
#     mujoco.mj_step(model, data)
#     if len(frames) < data.time * framerate:
#       renderer.update_scene(data)
#       pixels = renderer.render()
#       frames.append(pixels)

# media.show_video(frames, fps=framerate)



import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('/path/to/mjcf.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)