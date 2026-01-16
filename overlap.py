import torch
import time

# 1. 单独计算（矩阵乘法）
def compute_time(gpu_tensor):
    assert gpu_tensor.is_cuda
    total  = 0
    for _ in range(10):
        start = time.time()
        _ = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()  # 确保计算完成
        total += (time.time() - start)
    return total /10


# 2. 单独通信（CPU->GPU传输）
def communicate_time(cpu_tensor):
    total = 0
    repeat = 10
    for _ in range(repeat):
        start = time.time()
        _ = cpu_tensor.to("cuda")
        torch.cuda.synchronize()
        total += (time.time() - start)
    
    return total / repeat


# 3. 计算与通信 overlap 测试
# 同时启动通信和计算
def overlap_time(gpu_tensor, cpu_tensor, stream=False):
    assert gpu_tensor.is_cuda
    total = 0
    cp_stream = torch.cuda.Stream()
    for _ in range(10):
        start = time.time()
        if stream:
            with torch.cuda.stream(cp_stream):
                comm_op = cpu_tensor.to("cuda", non_blocking=True)  # 异步传输
        else:
            comm_op = cpu_tensor.to("cuda", non_blocking=True)  # 异步传输
        _ = torch.matmul(gpu_tensor, gpu_tensor)             # 同时计算
        torch.cuda.synchronize()  # 等待全部完成
        total += ( time.time() - start)
    return total/10

# 输出结果

if __name__ == "__main__":
    # 设置设备
    assert torch.cuda.is_available() 

    # 生成测试数据
    size = 10000 * 2
    size1 = size * 10
    cpu_tensor = torch.randn(size1, size, device = "cpu", pin_memory=True)
    gpu_tensor = torch.randn(size, size, device="cuda")
    _ = torch.matmul(gpu_tensor, gpu_tensor)

    torch.cuda.synchronize()

    compute = compute_time(gpu_tensor)
    comm = communicate_time(cpu_tensor)
    overlap_nostream = overlap_time(gpu_tensor, cpu_tensor)
    overlap_stream = overlap_time(gpu_tensor, cpu_tensor, stream=True)
    
    print(f"Compute time:               {compute:.4f} seconds")
    print(f"Communication time:         {comm:.4f} seconds")
    print(f"Overlap time(stream on):    {overlap_stream:.4f} seconds")
    print(f"Overlap time(stream off):   {overlap_nostream:.4f} seconds")
    print(f"Theoretical minimum:        {max(compute, comm):.4f} seconds")