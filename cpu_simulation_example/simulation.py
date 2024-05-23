import torch
import cpu
import numpy as np

from draw import Draw_MPC_point_stabilization_v1

# xx = []
empty_tensor = torch.empty(0)
empty_tensor = empty_tensor.type(torch.DoubleTensor)
position = cpu.cpu(empty_tensor)



tensor_reduced = position[:, :3]
mask = ~(tensor_reduced == 0).all(dim=1)
tensor_filtered = tensor_reduced[mask]
tensor_list = tensor_filtered.tolist()

print("tensor_list.shape = " , np.array(tensor_list).shape)
print("tensor_list = " , tensor_list)




x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)# initial state
xs = np.array([0, 4, 0.0]).reshape(-1, 1) # final state

draw_result = Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0, target_state=xs, robot_states=tensor_list )

