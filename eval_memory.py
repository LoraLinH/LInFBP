import os
import numpy as np
from scipy.signal.windows import hann, cosine
import torch 
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import time
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq, fftshift
from Utils.initParameter import InitPara
from Model.model_fbp import FBP
from Model.model_fbp_nearest import FBP_Nearest
from Model.model_fbp_cubic import FBP_Cubic
from Model.model_fbp_L import FBP_L
from Model.model_fbp_F import FBP_F
from Model.iRadonMap_Net import iRadonMap
from Model.iRadonMap_Net_L import iRadonMap_L
from Model.iRadonMap_Net_F import iRadonMap_F
from Model.DICDNet import DICDNet
from Model.DICDNet_F import DICDNet_F
from Model.DICDNet_L import DICDNet_L
from thop import profile, clever_format


def evaluate_model(model_name, model, input_tensor, device="cuda", loops=100):
    """
    计算模型的 FLOPs, 参数量, 显存占用, 推理时间
    """
    results = {}

    if 'DICD' in model_name:
        model.for_flops = True

    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    print(f"正在测评模型: {model_name} ...")

    # ---------------------------
    # A. 计算参数量 (Params)
    # ---------------------------
    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results["Params (M)"] = num_params / 1e6  # 转换为百万 (Million)
    except Exception as e:
        results["Params (M)"] = 0
        print("非学习型算法")

    # ---------------------------
    # B. 计算 FLOPs (MACs)
    # ---------------------------
    try:
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
        results["FLOPs (G)"] = macs / 1e9  # 转换为 Giga
    except Exception as e:
        results["FLOPs (G)"] = 0
        print(f"  [Warning] 无法计算 FLOPs (可能包含不支持的算子): {e}")

    # ---------------------------
    # C. 计算显存占用 (Max Memory)
    # ---------------------------
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # 重置计数器

    with torch.no_grad():
        # 运行一次前向传播
        _ = model(input_tensor)

    # 获取最大显存占用
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转换为 MB
    results["Max Memory (MB)"] = max_memory

    # ---------------------------
    # D. 计算推理时间 (Inference Time)
    # ---------------------------
    # 1. Warm-up (预热): GPU 需要预热以达到稳定状态
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # 2. 计时
    torch.cuda.synchronize()  # 等待所有 GPU 任务完成
    start_time = time.time()

    with torch.no_grad():
        for _ in range(loops):
            _ = model(input_tensor)
            # 如果中间有异步操作，这里不需要 sync，只在最后 sync 即可

    torch.cuda.synchronize()  # 等待循环结束
    end_time = time.time()

    avg_time = (end_time - start_time) / loops
    results["Inference Time (ms)"] = avg_time * 1000  # 转换为毫秒
    results["FPS"] = 1.0 / avg_time  # 每秒帧数

    return results

def main():

    opt = InitPara()

    geo = {'nVoxelX': 512, 'sVoxelX': 340.0192, 'dVoxelX': 0.6641,
                'nVoxelY': 512, 'sVoxelY': 340.0192, 'dVoxelY': 0.6641,
                'nDetecU': 736, 'sDetecU': 0.6848*2*736, 'dDetecU': 0.6848*2,
                'offOriginX': 0.0, 'offOriginY': 0.0,
                'views': 100, 'slices': 1,
                'DSD': 1085.6, 'DSO': 595.0, 'DOD': 490.6,
                'start_angle': 0.0, 'end_angle': 2*np.pi,
                'mode': 'fanflat', 'extent': 1, # currently extent supports 1, 2, or 3.
                }

    w = (geo['nDetecU'] - 1) / 2
    s = geo['dDetecU'] * (np.arange(geo['nDetecU']) - w)
    gam = np.arctan(s / geo['DSD'])
    w1 = np.abs(geo['DSO'] * np.cos(gam) - 0 * np.sin(gam)) / geo['DSD']
    geo['w1'] = torch.from_numpy(w1).cuda()

    npad = 2 ** np.ceil(np.log2(2 * geo['nDetecU'] - 1))  # padded size
    npad = int(npad)
    nnp = np.arange(-(npad // 2), npad // 2)
    h = np.zeros_like(nnp, dtype=float)
    h[npad // 2] = 1 / 4
    odd = nnp % 2 == 1
    h[odd] = -1 / (np.pi * nnp[odd]) ** 2
    h /= geo['dDetecU'] ** 2
    Hk = np.real(fft(fftshift(h)))

    window = np.ones((npad))
    # window = hann(npad)
    window = fftshift(window)
    Hk = Hk * window
    geo['filter'] = torch.from_numpy(Hk * geo['dDetecU']).cuda()

    betas = np.linspace(geo['start_angle'], geo['end_angle'], geo['views'], False)
    betas = np.expand_dims(np.expand_dims(betas, 0), 0)
    xc = np.arange(1, geo['nVoxelX'] + 1) - (geo['nVoxelX'] + 1) / 2
    yc = np.arange(1, geo['nVoxelY'] + 1) - (geo['nVoxelY'] + 1) / 2
    yc = np.flip(yc)
    xc = np.expand_dims(np.expand_dims(xc, -1), 0) * geo['dVoxelX']
    yc = np.expand_dims(np.expand_dims(yc, -1), -1) * geo['dVoxelY']
    d_loop = geo['DSO'] - xc * np.sin(betas) + yc * np.cos(betas)  # dso - y_beta
    mag = geo['DSD'] / d_loop
    geo['w2'] = torch.from_numpy(mag ** 2).cuda()  # [np] image-domain weighting

    geo['indices'] = torch.abs(100 * torch.rand((geo['nVoxelX']*geo['nVoxelY']*geo['views']))).cuda()

    input_tensor = torch.randn(1, 1, geo['views'], geo['nDetecU'])

    models_dict = {
        "model_fbp_nearest": FBP_Nearest(geo), # 示例：取消注释并填入你的实例
        "model_fbp_linear": FBP(geo),
        "model_fbp_cubic": FBP_Cubic(geo),
        "model_fbp_L": FBP_L(geo),
        "model_fbp_F": FBP_F(geo),
        "iRadonmap": iRadonMap(geo, opt),
        "iRadonmap_L": iRadonMap_L(geo, opt),
        "iRadonmap_F": iRadonMap_F(geo, opt),
        "DICDNet": DICDNet(geo),
        "DICDNet_L": DICDNet_L(geo),
        "DICDNet_F": DICDNet_F(geo),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_metrics = []

    for name, model in models_dict.items():
        metrics = evaluate_model(name, model, input_tensor, device=device, loops=50)
        metrics["Model"] = name
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    # 调整列顺序
    cols = ["Model", "Params (M)", "FLOPs (G)", "Max Memory (MB)", "Inference Time (ms)", "FPS"]
    df = df[cols]

    print("\n" + "=" * 50)
    print("最终测评结果")
    print("=" * 50)
    print(df.to_string(index=False))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    main()

