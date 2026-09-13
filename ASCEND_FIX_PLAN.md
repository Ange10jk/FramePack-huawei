# FramePack-huawei 算子移植审查

审查日期：2026-09-13。服务器目录：`/mnt/luojk/pipeline/FramePack-huawei`。对照目录：`/mnt/luojk/pipeline/FramePack`。

## 范围与结论

逐一比较并阅读项目的全部 17 个 Python 源文件，另检查 Ascend 依赖清单、实际安装的 HunyuanVideo VAE 因果卷积实现和本地模型缓存。未扫描第三方依赖的全部实现或模型权重数值。17 个文件均通过 AST 语法检查。执行小张量 CPU/NPU 对照，不加载视频大模型，没有运行完整视频生成，也没有修改推理源码或提交 Git。

存在确定的移植和调度错误，尤其是默认视频时长触发 CPU VAE、高显存模式的 Transformer 卸载后未恢复，以及 F1 入口遗漏兼容处理。核心注意力及位置编码降采样的小张量结果暂未显示会直接造成纯噪点的明显错误。不能据此宣布噪点根因已定位。

## 优先修复项

### 1. [P1] 默认 5 秒视频会强制转到 CPU FP32 VAE

位置：`/mnt/luojk/pipeline/FramePack-huawei/demo_gradio.py:295`，CPU 分支在 302–303 行。

只有 `total_second_length <= 1` 且 `FRAMEPACK_VAE_DEVICE=npu` 才走 NPU 解码。UI 默认时长为 5 秒，因此即使配置 npu，也会把完整 VAE 移到 CPU 并执行 `vae.float()`。这造成较大的 RAM/计算负担，与用户指定的 NPU 解码路径不符。应以明确的设备策略决定 VAE 位置，不以视频时长覆盖配置。

此外，第 63 行根据 CPU 配置选择 FP32 加载，但第 85 行又无条件转回 FP16，说明 dtype 配置也不一致。此前“默认仍使用 NPU VAE”的答复不准确。

### 2. [P1] 高显存模式下 Transformer 被卸载后没有重新加载

位置：`/mnt/luojk/pipeline/FramePack-huawei/demo_gradio.py:290`；加载条件见第 226 行。

每段采样后都执行 `transformer.to(cpu)`，但下段采样前的加载仅在 `not high_vram` 时发生。高显存模式没有安装 DynamicSwap，因此后续段会出现 CPU 权重与 NPU 输入混用。即使第一段测试只有一秒，第二次生成也可能遇到这一状态。

本次读取 npu:0 可用显存约 60.397 GiB，超过代码的 60 GiB 阈值；这个分支在当前空闲服务器上实际可触发。修复时必须使卸载、迁回与 high_vram 状态一致。

### 3. [P1] CPU 实验路径把全局 Transformer 清为 None，破坏后续任务

位置：`/mnt/luojk/pipeline/FramePack-huawei/demo_gradio.py:294`。

配置 `FRAMEPACK_VAE_DEVICE=cpu` 时，最后一个 section 将全局 `transformer` 置为 None，但之后没有重新构造模型。下次请求访问 `transformer.dtype` 或卸载它会失败；低显存模式的异常清理还会再次对 None 调用 `.to()`，从而可能跳过最终 end 队列事件，导致界面一直等待。

应实现完整的销毁与重新加载机制，或者保留可复用模型实例并只管理其设备位置；不应直接破坏全局服务状态。

### 4. [P1] F1 入口漏装 BF16 replicate padding 兼容处理

位置：`/mnt/luojk/pipeline/FramePack-huawei/demo_gradio_f1.py:22`；实际调用：`/mnt/luojk/pipeline/FramePack-huawei/diffusers_helper/models/hunyuan_video_packed.py:83`。

主入口安装 npu_compat，F1 入口仅导入 device。F1 同样采用 BF16 Transformer，其 clean_latents_2x/4x 路径调用 5D replicate padding。本次实测当前 CANN 原生该算子支持 FP16/FP32，但 BF16 报 `self not implemented for DT_BFLOAT16`。

应统一入口初始化或将兼容实现局部应用到共享算子调用处。不能把“FP16 VAE padding 可运行”推论为“BF16 Transformer padding 也可运行”。

### 5. [P1] F1 预览仍把 FP32 Tensor 送入不支持的 NPU Conv3D

位置：`/mnt/luojk/pipeline/FramePack-huawei/demo_gradio_f1.py:214`；被调用位置：`/mnt/luojk/pipeline/FramePack-huawei/diffusers_helper/hunyuan.py:87`。

采样器返回的预览 latent 是 FP32。F1 直接调用 `vae_decode_fake(preview)`，该函数在输入设备和 dtype 上构造权重后执行 conv3d。本次小张量实测 NPU FP32 Conv3D 报 EZ3002，支持列表包含 FP16/BF16 而不包含 FP32。主入口第 237 行已通过 `preview.cpu()` 避开此问题，但 F1 未同步。

应统一预览执行策略，或使用验证过的等价投影实现。预览异常不应终止视频采样。

### 6. [P2] 全局 padding 补丁不支持负 padding，返回错误形状

位置：`/mnt/luojk/pipeline/FramePack-huawei/diffusers_helper/diffusers_helper/npu_compat.py:15`。

补丁把所有非零 before/after 当成扩展长度，未处理负数表示裁剪的语义。对 `[0,1,2,3,4,5]` 使用 `pad=(-1,0,0,0,0,0)`，正确结果应为 `[1,2,3,4,5]`；实测返回 `[0,0,1,2,3,4,5]`，长度从 6 增至 7。

当前 FramePack 已检查的调用采用非负 padding，因此该错误不是已证实的当前噪点原因。由于补丁全局替换 F.pad，应限制补丁适用范围或正确实现负 padding。

### 7. [P2] 模型迁到 CPU 不会迁移或清除 TeaCache 普通 Tensor 属性

位置：`/mnt/luojk/pipeline/FramePack-huawei/diffusers_helper/models/hunyuan_video_packed.py:1020` 和第 1051 行；卸载点：`/mnt/luojk/pipeline/FramePack-huawei/demo_gradio.py:293`。

`previous_modulated_input`、`previous_residual` 是普通 Tensor 属性，没有注册为 parameter/buffer。实测 `.cpu()` 后，模型权重在 CPU，而这类属性仍在 npu:0。原版也存在缓存保留逻辑，因此这属于沿用行为与当前卸载方案不匹配，不能全部归因为新移植改动。

section 结束、进入 VAE 解码前应显式清除不再需要的缓存。该事实能解释“权重搬走后显存仍未完全下降”的一部分，不能证明它就是所有 OOM 的主因。

## 逐文件检查表

下表路径相对于 `/mnt/luojk/pipeline/FramePack-huawei`。相同表示与服务器同级官方目录逐文件 diff 相同；不代表对全部输入、精度和硬件均已验证。

| 文件 | 审查结果 |
| --- | --- |
| demo_gradio.py | 确定存在问题 1–3；CPU VAE dtype 配置被后续覆盖；高显存模式需重点修复。 |
| demo_gradio_f1.py | 确定存在问题 4–5；另外缺少主入口的 worker 内 NPU set_device，非默认设备线程上下文需单独验证。 |
| diffusers_helper/device.py | 自动设备选择和指定 NPU 路径未发现当前单卡静态错误；跨线程、多后端组合未全面验证。 |
| diffusers_helper/diffusers_helper/npu_compat.py | 非负 padding 在 FP16/BF16/FP32 小张量上与 CPU 参考一致；负 padding 错误，见问题 6。无 padding 的维度也 cat，会产生不必要的中间分配。 |
| diffusers_helper/models/hunyuan_video_packed.py | attention 布局和 scale 未发现明显错误；pool 替代小测试通过。存在问题 7。varlen 仅对当前相同 Q/KV 长度的联合自注意力路径做了验证，不是通用 cu_seqlens 实现。 |
| diffusers_helper/memory.py | 当前 NPU 的 active/reserved keys 存在，空闲显存读数正常。unload_complete_models 将模型移到 CPU，不代表释放 RAM 或普通缓存属性；不能接收 None。 |
| diffusers_helper/hunyuan.py | VAE latent scaling 与原版一致。新增 `.cpu()` 仅在 image_mode=True；主视频默认 False，因此这项改动并没有实现主视频分块解码。FP32 fake preview 在 NPU 上不兼容，见问题 5。 |
| diffusers_helper/bucket_tools.py | 新增低分辨率 bucket 保持 16 倍数，未发现明确布局错误；64/128 不是原版正常画质验收尺寸。 |
| diffusers_helper/dit_common.py | 与原版相同；归一化 FP32 计算/回转逻辑未发现新增移植错误。未对完整模型层逐一数值对照。 |
| diffusers_helper/pipelines/k_diffusion_hunyuan.py | 与原版相同；sigma、guidance、CPU generator 后迁 NPU 的路径未发现新增改错。 |
| diffusers_helper/k_diffusion/uni_pc_fm.py | 与原版相同；25 步 toy 模型 CPU/NPU 对照通过。linalg.solve 会回退 CPU，是小矩阵求解，区别于整个 VAE 在 CPU 上执行。 |
| diffusers_helper/k_diffusion/wrapper.py | 与原版相同；flow matching 符号、timestep scale、CFG 组合未发现新增错误。 |
| diffusers_helper/clip_vision.py | 与原版相同；输入预处理明确迁至 image_encoder 的 device/dtype。 |
| diffusers_helper/utils.py | 只有显存查询工具改为通用后端；视频 RGB 布局、[-1,1] 到 uint8 及 libx264 写出保持原版。未发现新增视频编码错误。 |
| diffusers_helper/thread_utils.py | 与原版相同；使用 threading.Thread 和列表队列，不是 multiprocessing。不能把此前 multiprocessing EOFError 直接归因于该文件。 |
| diffusers_helper/hf_login.py | 与原版相同；无算子移植内容。 |
| diffusers_helper/gradio/progress_bar.py | 与原版相同；无算子移植内容。 |

## 小张量实测证据

环境实际导入版本：torch 2.3.1、torch_npu 2.3.1。使用 npu:0；测试不加载模型权重。

| 测试 | 结果 |
| --- | --- |
| 补丁非负 replicate padding，3 种 dtype、3 组 padding | 与量化至相同 dtype 的 CPU 参考最大误差 0。 |
| unfold/mean 对比 avg_pool3d，kernel 2 和 4 | 最大绝对误差分别约 8.94e-8、5.96e-8。 |
| NPU fusion attention，head_dim=128，seq=32/79，FP16 | 最大绝对误差约 5.1e-4；有限值，fusion 分支未回退。 |
| 同上，BF16 | 最大绝对误差约 3.8e-3；有限值。仅验证小规模输入。 |
| batch=2，右 padding 的 varlen 有效 token | CPU fallback 与逐样本参考最大误差 0；cu_seqlens=[0,6,8,15,16]。 |
| 原生 temporal replicate padding，3 组 shape | FP16/FP32 通过；BF16 均报不支持。 |
| 原版 UniPC，25 步 toy model | CPU/NPU 最大绝对误差约 3.73e-9；输出有限；solve 自动回退 CPU。 |
| NPU FP32 1×1×1 Conv3D | 明确报不支持 DT_FLOAT。 |
| 新进程 FP16/BF16 Conv3D | 输出均有限；相对于 CPU FP32 参考最大误差约 0.00380/0.02753。 |
| 负 padding | 可稳定复现错误长度与数值。 |
| 普通 Tensor 属性经 Module.cpu() | 仍在 npu:0。 |

说明：包含 FP32 Conv3D 底层错误的进程，其后续 FP16 测试曾出现非有限值；该结果已排除，不作为正常 FP16 算子错误的证据。使用独立新进程复核后，FP16/BF16 输出均正常。

注意力的 BSND 布局受官方接口支持，见 [Ascend npu_fusion_attention 文档](https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000850.html)。本报告的数值判断以当前服务器实测为依据。

## 纠正此前缓存诊断

本次核查 text_encoder 有 6 个文件、索引中 4 个权重分片均存在；text_encoder_2、tokenizer、tokenizer_2、vae 检查未发现断开的一级文件链接。未进行全量权重校验和验证或完整离线模型加载。

直接调用 try_to_load_from_cache 的结果：

- cache_dir=/mnt/luojk/pipeline/FramePack-huawei/hf_download：返回 None。
- cache_dir=/mnt/luojk/pipeline/FramePack-huawei/hf_download/hub：正确返回 text_encoder/config.json。

此前 direct9 将 HF_HUB_CACHE 指向 hf_download，少了一层 hub。已证实存在缓存目录配置错误，不能据此前离线加载报错宣称必须重新下载文本编码器。保留 HF_HOME=hf_download 时，通常应让 hub 缓存使用其下的 hub 目录，或显式把 HF_HUB_CACHE 指到该目录。

现有 ASCEND_FIX_PLAN.md 将“缺失模型”列为确定事实，该项需要按本报告纠正；随机 latent 解码的验收也应比较 CPU/NPU一致性和有限值，不能要求随机 latent 本身解码成正常语义画面。

## 其他发现及验证边界

- requirements-ascend-lock.txt 包含 /root/selfgz... 和 /usr/local/Ascend/... 的 file:// wheel 引用，属于环境导出快照，不能直接视为可移植安装锁文件。
- fusion attention 的 RuntimeError 捕获会永久关闭全局 fusion 标志。需区分不支持的 shape 与严重设备错误；当前小测试未触发 fallback，不能断言 fallback 是当前 OOM 根因。
- 目前只做小张量验证；长序列 attention、真实权重加载完整性、完整 VAE 输出、连续多 section 推理和最终视频画质仍未验收。
- 本次 NPU 编译可能生成/更新 kernel_meta、ge_check_op.json、fusion_result.json 等诊断产物；没有修改项目 Python 源码。

推荐先修复主入口调度问题 1–3，再统一 F1 兼容路径并处理缓存生命周期，之后用正常步数、固定输入验证真实 latent 和解码帧。不能把当前所有噪点归因于一个尚未复现的算子错误。
