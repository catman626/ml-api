import torch
import torch.profiler

# 准备数据：在CPU上创建一个大张量
cpu_tensor = torch.randn(10000, 10000)

# 启动Profiler，必须同时开启CPU和CUDA活动监控
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,  # 必须开启才能看到GPU上的传输内核
    ],
    # 可选：将结果记录到文件，方便用TensorBoard查看
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/trace_transfer'),
    record_shapes=True,  # 记录张量形状，方便分析
    profile_memory=False,  # 如果不需要内存分析可以关闭以减少开销
    with_stack=False
) as prof:
    
    # --- 开始测量具体操作 ---
    
    # 1. CPU -> GPU 传输
    with torch.profiler.record_function("Memcpy_H2D"): # Host to Device
        # 先同步，确保之前的计算都完成了
        torch.cuda.synchronize() 
        # 记录开始时间（CPU时间）
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        # 执行传输
        gpu_tensor = cpu_tensor.cuda(non_blocking=False) # 阻塞模式便于测量
        
        end.record()
        torch.cuda.synchronize() # 必须同步，否则时间测不准
    
    # 2. GPU -> CPU 传输
    with torch.profiler.record_function("Memcpy_D2H"): # Device to Host
        torch.cuda.synchronize()
        start.record()
        
        # 执行传输
        final_tensor = gpu_tensor.cpu() # 这会触发同步
        
        end.record()
        torch.cuda.synchronize()

    # --- 结束测量 ---

# 打印关键摘要（可选）
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 保存跟踪结果
prof.export_chrome_trace("trace_transfer.json")