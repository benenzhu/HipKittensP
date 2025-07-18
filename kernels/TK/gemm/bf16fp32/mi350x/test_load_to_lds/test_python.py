import torch
import tk_kernel
import random

profiling = True
profiling_ref = False
torch.manual_seed(0)
random.seed(0)

# Inputs
ROWS = 64
COLS = 64

A = torch.randn(ROWS, COLS, dtype=torch.bfloat16, device='cuda') / 10.0  
C = torch.zeros(ROWS, COLS, dtype=torch.bfloat16, device='cuda')
C_ref = torch.zeros(ROWS, COLS, dtype=torch.bfloat16, device='cuda')  

tk_kernel.dispatch_micro(A, C, C_ref)

# C_ref = torch.matmul(A, A.t()).float()
C_ref = A.float()

print("Out")
print(C[:, 0:8])
print("Ref")
print(C_ref[:, 0:8])

diff = C.float() - C_ref.float()
# print(f"diff[0:4]")
# print(diff[0:4])

# print(f"diff[4:8]")
# print(diff[4:8])

# print(f"diff[8:12]")
# print(diff[8:12])

# print()
# print(f"diff[12:16, 0:8]")
# print(diff[12:16, 0:8])

# print()
# print(f"diff[16:20, 0:8]")
# print(diff[16:20,0:8])


max_diff = diff.abs().max()
print(f"Max diff: {max_diff}")