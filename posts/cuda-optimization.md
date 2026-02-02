---
title: CUDA 内存优化实战
date: 2024-01-20
tags: [cuda, optimization, memory]
category: 性能优化
description: 深入理解 CUDA 内存层次结构，掌握共享内存、常量内存和全局内存的优化技巧。
---

# CUDA 内存优化实战

内存访问是 GPU 编程中最关键的性能因素之一。本文将深入探讨 CUDA 的内存层次结构，并提供实用的优化技巧。

## CUDA 内存层次结构

```
┌─────────────────────────────────────────────────────────┐
│                    寄存器 (Registers)                    │
│                    - 最快存储                           │
│                    - 每个线程私有                        │
├─────────────────────────────────────────────────────────┤
│                   共享内存 (Shared Memory)               │
│                    - 线程块内共享                        │
│                    - 可配置大小                          │
├─────────────────────────────────────────────────────────┤
│                  常量内存 (Constant Memory)              │
│                    - 只读存储                            │
│                    - 广播读取                            │
├─────────────────────────────────────────────────────────┤
│                  全局内存 (Global Memory)                │
│                    - 最大容量                            │
│                    - 需要显式管理                        │
└─────────────────────────────────────────────────────────┘
```

## 全局内存优化

### 合并内存访问

```cuda
// 不优化的随机访问
__global__ void bad_kernel(float* data, int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 非合并访问 - 性能差
        float value = data[indices[idx]];
        data[idx] = value * 2.0f;
    }
}

// 优化的合并访问
__global__ void good_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 合并访问 - 相邻线程访问相邻内存
        float value = input[idx];
        output[idx] = value * 2.0f;
    }
}
```

### 内存对齐

确保数据按 128 字节或 256 字节对齐：

```cuda
// 使用 cudaMalloc 分配已对齐的内存
float* d_data;
cudaMalloc(&d_data, size * sizeof(float));

// 使用 cudaMemcpy 异步传输
cudaMemcpyAsync(d_data, h_data, size * sizeof(float),
                cudaMemcpyHostToDevice, stream);
```

## 共享内存优化

共享内存是 GPU 上最快的可编程内存，适合存储频繁访问的数据。

### 矩阵乘法优化

```cuda
#define TILE_SIZE 16

__global__ void tiled_matrix_mul(float* A, float* B, float* C,
                                  int M, int N, int K) {
    // 声明共享内存
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 遍历所有瓦片
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // 加载 A 的瓦片
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载 B 的瓦片
        if (col < N && tile * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 同步确保数据加载完成
        __syncthreads();

        // 计算当前瓦片的部分和
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // 同步避免共享内存覆盖
        __syncthreads();
    }

    // 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 避免银行冲突

共享内存被组织为多个银行，需要注意访问模式：

```cuda
// 避免连续线程访问连续银行（银行冲突）
// 不好的模式
__shared__ float data[256];
float val = data[threadIdx.x];  // 可能导致银行冲突

// 好的模式：使用交错访问
float val = data[threadIdx.x * 4 % 256];
```

## 常量内存优化

常量内存适合存储核函数中只读的数据：

```cuda
// 声明常量内存
__constant__ float coefficients[256];

// 从主机端设置常量内存
cudaMemcpyToSymbol(coefficients, h_coeffs, sizeof(h_coeffs));

// 核函数中使用
__global__ void use_constants(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        // 常量内存读取自动广播到所有线程
        val *= coefficients[idx % 256];
        output[idx] = val;
    }
}
```

## 内存优化最佳实践

### 1. 最大化内存带宽利用率

```cuda
// 使用向量化加载
int4* data = (int4*)d_array;
int4 val = data[threadIdx.x];  // 一次加载 4 个 int

// 使用异步内存传输和计算重叠
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 第一个流的计算
kernel1<<<grid, block, 0, stream1>>>(d_data1);

// 第二个流的数据传输和计算
cudaMemcpyAsync(d_data2_host, d_data2_dev, size, cudaMemcpyDeviceToHost, stream2);
kernel2<<<grid, block, 0, stream2>>>(d_data2_dev);
```

### 2. 合理使用内存类型

| 场景 | 推荐内存类型 |
|------|-------------|
| 线程私有数据 | 寄存器 |
| 线程块内共享数据 | 共享内存 |
| 全局只读数据（小于 64KB） | 常量内存 |
| 大规模只读数据 | 只读全局内存 (LDG) |
| 频繁读写数据 | 寄存器 > 共享内存 > 全局内存 |

### 3. 内存预取

```cuda
__global__ void prefetch_kernel(float* data, int n) {
    __shared__ float tile[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 预取下一块数据
    int next_row = (ty + 32) % (n / 32);

    // 当前数据
    for (int i = 0; i < n / 32 / 32; i++) {
        // 异步预取下一行（可选）
        // __ldg() 用于只读数据的缓存读取
        float val = __ldg(&data[(ty + i * 32) * 32 + tx]);

        tile[ty][tx] = val;
        __syncthreads();

        // 处理当前瓦片
        // ...

        __syncthreads();
    }
}
```

## 性能分析工具

使用 `nvprof` 或 Nsight Systems 分析内存性能：

```bash
# 分析内存传输
nvprof --metrics gld_efficiency,gst_efficiency ./program

# 分析内存访问模式
nvprof --print-gpu-trace ./program

# 使用 Nsight Compute
nsight compute ./program
```

## 总结

内存优化是 CUDA 性能提升的关键：

1. **合并访问**：确保相邻线程访问相邻内存
2. **充分利用共享内存**：减少全局内存访问
3. **避免银行冲突**：优化共享内存访问模式
4. **异步传输**：隐藏内存延迟
5. **性能分析**：使用工具指导优化方向
