import torch
import graph_max

test_input = torch.randn([1,2]).cuda()
# print("test_input = " , test_input)
max_val = graph_max.graph_max(test_input)
print(max_val)
