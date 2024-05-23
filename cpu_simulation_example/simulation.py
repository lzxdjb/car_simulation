import torch
import cpu
import numpy as np

from draw import Draw_MPC_point_stabilization_v1

# xx = []
empty_tensor = torch.empty(0)
max_val = cpu.cpu(empty_tensor)
# print("max_val = " , max_val)
tensor_list = max_val.tolist()
# print("tensor_list = " , tensor_list)



x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)# initial state
xs = np.array([1.5, 1.5, 0.0]).reshape(-1, 1) # final state

# draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0, target_state=xs, robot_states=xx )

