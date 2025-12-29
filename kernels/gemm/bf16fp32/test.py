import torch
import tk_kernel
import random
from utils import init_randn, init_empty

test_shapes = [
    (256, 256, 128), # (m, n, k)
    (4096, 8192, 2048),
    (8192, 4096, 2048),
    (8192, 2048, 4096),
    (512, 1024, 1024),
    (512, 1024, 2048),
    (2048, 1024, 512),
]

torch.manual_seed(0)
random.seed(0)
dtype = torch.bfloat16
device = "cuda:0"

from triton_matmul import matmul

if __name__ == "__main__":
    for test_shape in test_shapes:
        m, n, k = test_shape
        A = torch.arange(m * k).reshape(m, k).cuda().bfloat16() * 0.01
        B = torch.arange(k * n).reshape(k, n).cuda().bfloat16() * 0.01
        Bt = B.t().contiguous()
        C = init_empty((m, n), dtype, device)

        C_ref = matmul(A, B)
        torch_ref = matmul(A, B)
        tk_kernel.dispatch_micro(A, Bt, C)

        is_valid = torch.allclose(C, C_ref, rtol=1e-2)
        torch.cuda.synchronize()
        exit(0)
        result = "TEST PASSED" if is_valid else "TEST FAILED"
        assert is_valid, f"{C=} {C_ref=} {C-C_ref} {torch_ref - C_ref}"
        print(f"{test_shape}".ljust(18) + f" | {result}")