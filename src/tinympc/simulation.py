import torch
import simulation

test_input = torch.randn([1,2]).cuda()
# print("test_input = " , test_input)
max_val = simulation.simulation(test_input)
print(max_val)