---
title: Triton 入门指南
date: 2024-01-15
tags: [triton, tutorial, gpu]
category: 基础教程
description: Triton 编程语言的基础概念、语法和使用方法，帮助你快速上手 GPU 编程。
---

# Triton 入门指南

Triton 是 OpenAI 开源的 GPU 编程语言，旨在让开发者能够更高效地编写高性能 CUDA 内核。本文将介绍 Triton 的基础概念和使用方法。

## 为什么选择 Triton？

传统的 GPU 编程需要深入了解 CUDA 架构和硬件细节，开发门槛较高。Triton 提供了以下优势：

- **Python 语法**：使用 Python 编写，降低学习成本
- **自动优化**：Triton 编译器自动处理内存访问和并行化
- **灵活调度**：细粒度的线程块和网格调度控制
- **易于调试**：支持 Python 调试器和断点

## 环境安装

首先安装 Triton：

```bash
pip install triton
```

验证安装：

```python
import triton
print(triton.__version__)
```

## 第一个 Triton 内核

让我们编写一个简单的向量加法内核：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 获取线程块 ID 和线程 ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # 创建线程偏移量
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建掩码，只处理有效元素
    mask = offsets < n_elements

    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 执行加法
    output = x + y

    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 内核调用

使用 `triton.jit` 装饰的函数可以通过 `kernel[grid](args)` 的方式调用：

```python
import torch

# 创建输入张量
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
c = torch.empty(1024, device='cuda')

# 定义网格大小
grid = (1024,)

# 调用内核
add_kernel[grid](a, b, c, 1024, 128)
```

## 常见模式

### 1. 二维索引

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak,
                  stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr):
    # ... 矩阵乘法实现
```

### 2. 归约操作

```python
@triton.jit
def sum_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # ... 归约实现
```

## 性能调优技巧

1. **选择合适的 Block Size**：通常 128 或 256 是好的起点
2. **使用掩码**：确保内存访问不越界
3. **利用向量化加载**：使用 `tl.load` 的向量化版本
4. **避免分支分歧**：同一线程束内的线程应执行相同代码

## 总结

Triton 为 GPU 编程提供了更高级别的抽象，同时保留了足够的灵活性进行性能优化。通过本文的学习，你应该能够：

- 安装和配置 Triton 开发环境
- 编写基本的 Triton 内核
- 理解 Triton 的编程模型和最佳实践

## 参考资源

- [Triton 官方文档](https://triton-lang.org/)
- [Triton GitHub 仓库](https://github.com/openai/triton)
- [Triton 教程](https://triton-lang.org/getting-started/tutorials.html)
