# RTX 2080 Ti (Turing sm_75) GPU 兼容性适配计划

## 背景

RTX 2080 Ti 为 Turing 架构，Compute Capability 7.5，主要限制：
- **不支持 BF16** 硬件加速
- **Flash Attention** 需编译时指定 `CUDA_ARCH=75`，且官方支持从 sm_75 起才引入
- CUDA 11.8 + PyTorch 2.5.1 组合在 Turing 上存在已知 Flash Attention 兼容问题

---

## 问题清单

### 🔴 严重（会导致报错/崩溃）

#### T1. `modules/attention.py` 顶层无条件导入 flash_attn

- **位置**: `model/vecformer/modules/attention.py:3`
- **问题**: `import flash_attn` 无 try-except 保护，若 flash_attn 未安装或编译不兼容 sm_75，直接 ImportError 导致整个模块无法加载
- **对比**: `point_transformer_v3/model.py:30-33` 已有 try-except 保护，风格不一致
- **修复方案**:
  ```python
  # 替换顶层硬导入为带 fallback 的条件导入
  try:
      import flash_attn
  except ImportError:
      flash_attn = None
  ```
- **影响类**: `VarlenSelfAttention`, `VarlenSelfAttentionWithRoPE`, `VarlenCrossAttention`（这三个类均无 `enable_flash` 开关，完全依赖 flash_attn）

#### T2. `VarlenSelfAttention` / `VarlenSelfAttentionWithRoPE` / `VarlenCrossAttention` 无非 Flash 回退路径

- **位置**:
  - `model/vecformer/modules/attention.py:39-44`（`VarlenSelfAttention.forward`）
  - `model/vecformer/modules/attention.py:130-135`（`VarlenSelfAttentionWithRoPE.forward`）
  - `model/vecformer/modules/attention.py:188-197`（`VarlenCrossAttention.forward`）
- **问题**: 三个类的 forward 直接调用 `flash_attn.flash_attn_varlen_*`，无 `enable_flash` 开关，无标准注意力回退路径
- **修复方案**: 为这三个类添加 `enable_flash` 参数，当 `flash_attn is None` 或 `enable_flash=False` 时切换到 `F.scaled_dot_product_attention` 或手动实现的变长注意力

---

### 🟠 高优先级（影响性能或数值稳定性）

#### T3. Flash Attention 路径中强制 `.half()` 转换

- **位置**:
  - `model/vecformer/modules/attention.py:40` — `qkv_packed.half()`
  - `model/vecformer/modules/attention.py:131` — `qkv_packed.half()`
  - `model/vecformer/modules/attention.py:189` — `q.half()`
  - `model/vecformer/modules/attention.py:190` — `kv[:, 0].half()`
  - `model/vecformer/modules/attention.py:191` — `kv[:, 1].half()`
  - `model/vecformer/point_transformer_v3/model.py:487` — `qkv.half().reshape(...)`
- **问题**: `.half()` 强制 float16，在 Turing 上 Flash Attention 若编译不正确会出现精度异常或 CUDA kernel 错误；另 `.half()` 写死了精度，不尊重上层 AMP 设置
- **修复方案**: 改为动态判断，仅在 Flash Attention 需要时做类型转换，转换后恢复原始 dtype（部分位置已有 `.to(inputs.dtype)` 恢复，但强制转换仍不合适）：
  ```python
  # 改为: 仅当输入不是 fp16 时才转换
  _dtype = qkv_packed.dtype
  if _dtype not in (torch.float16,):
      qkv_packed = qkv_packed.half()
  ```

#### T4. `configuration_vecformer.py` 默认开启 Flash Attention

- **位置**: `model/vecformer/configuration_vecformer.py:42`
- **问题**: `enable_flash=True` 是默认值，用户在 2080 Ti 上不做任何修改直接运行会触发 T1/T2/T3 的所有问题
- **修复方案**: 添加运行时 GPU capability 检测，自动决策是否启用 Flash Attention：
  ```python
  # 在 __init__ 中或单独工具函数里
  import torch
  def _auto_flash():
      if not torch.cuda.is_available():
          return False
      cc = torch.cuda.get_device_capability()
      # Flash Attention 官方要求 sm >= 7.5，但 Turing (7.5) 有已知问题
      # 保守起见，仅在 Ampere (sm_80+) 及以上默认启用
      return cc[0] > 7 or (cc[0] == 7 and cc[1] >= 5)  # 可调整阈值
  ```
  或者更保守地将默认值改为 `False`，由用户在高端卡上手动开启。

---

### 🟡 中优先级（依赖安装/文档）

#### T5. README 中 Flash Attention 编译命令未指定 `CUDA_ARCH`

- **位置**: `README.md:35`
- **当前**:
  ```bash
  MAX_JOBS=64 python setup.py install
  ```
- **问题**: 未指定目标架构，flash-attention 编译时可能不包含 sm_75 kernel，在 2080 Ti 运行时 fallback 到慢速实现或报错
- **修复方案**:
  ```bash
  # 2080 Ti (sm_75)
  CUDA_ARCH=75 MAX_JOBS=8 python setup.py install
  # 若同时需支持多架构:
  CUDA_ARCH="75;80;86" MAX_JOBS=8 python setup.py install
  ```

#### T6. `spconv-cu118==2.3.8` 对 Turing 支持需验证

- **位置**: `requirements.txt:7`，使用于 `point_transformer_v3/model.py:19`
- **问题**: spconv 二进制包通常针对特定 CUDA/架构编译，需确认 `spconv-cu118==2.3.8` 包含 sm_75 支持
- **验证方法**:
  ```bash
  python -c "import spconv; print(spconv.__version__)"
  # 运行一个简单 spconv 操作确认无 CUDA kernel 错误
  ```
- **备用方案**: 从源码编译 spconv 并指定 `TORCH_CUDA_ARCH_LIST="7.5"`

#### T7. `torch-scatter` 安装命令 PyTorch 版本号不一致

- **位置**: `README.md:29`
- **当前**:
  ```bash
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
  ```
- **问题**: 安装的 PyTorch 是 `2.5.1`，但 torch-scatter 下载页用的是 `torch-2.5.0`，可能导致 ABI 不匹配
- **修复方案**: 统一版本号为 `torch-2.5.1`：
  ```bash
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
  ```

---

### 🟢 低优先级（优化/增强）

#### T8. 缺少运行时 GPU 能力检查工具

- **问题**: 无任何启动时检测，用户不知道当前 GPU 是否支持所有特性
- **建议**: 在入口脚本或 `__init__.py` 中添加检测日志：
  ```python
  if torch.cuda.is_available():
      cc = torch.cuda.get_device_capability()
      if cc < (8, 0):
          warnings.warn(
              f"GPU compute capability {cc} < 8.0 (Ampere). "
              "Flash Attention may have issues. "
              "Consider setting enable_flash=False in backbone_config."
          )
  ```

---

## 修改优先级与工作量汇总

| ID  | 位置 | 类型 | 优先级 | 预估工作量 |
|-----|------|------|--------|------------|
| T1  | `modules/attention.py:3` | 代码修复 | 🔴 严重 | 5 min |
| T2  | `modules/attention.py` 三个类 forward | 代码修复（需实现回退路径） | 🔴 严重 | 2-4 h |
| T3  | `modules/attention.py` + `model.py` 共6处 | 代码修复 | 🟠 高 | 30 min |
| T4  | `configuration_vecformer.py:42` | 代码修复/配置 | 🟠 高 | 30 min |
| T5  | `README.md:35` | 文档修复 | 🟡 中 | 5 min |
| T6  | `requirements.txt:7` + spconv | 依赖验证 | 🟡 中 | 1 h |
| T7  | `README.md:29` | 文档修复 | 🟡 中 | 5 min |
| T8  | 入口脚本 | 功能增强 | 🟢 低 | 30 min |

---

## 快速临时解决方案（不改代码）

如需在 2080 Ti 上立即运行，手动修改配置文件关闭 Flash Attention：

```python
# 在你的训练/推理脚本里，初始化模型前覆盖配置
config.backbone_config["enable_flash"] = False
config.backbone_config["upcast_attention"] = True
config.backbone_config["upcast_softmax"] = True
```

> **注意**: 此方案仅对 `point_transformer_v3` 的骨干网络生效，
> `modules/attention.py` 中的 `VarlenSelfAttention` / `VarlenCrossAttention`
> 仍会尝试调用 flash_attn（T1/T2 问题），需 flash_attn 正确安装后才能使用。
