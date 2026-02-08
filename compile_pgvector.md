# macOS ARM64 编译 pgvector 完整指南

本文档介绍如何在 macOS ARM64 架构上编译和优化 pgvector 扩展（使用 NEON 指令集）。

## 目录

1. [系统要求](#系统要求)
2. [安装依赖](#安装依赖)
3. [编译步骤](#编译步骤)
4. [测试安装](#测试安装)
5. [性能验证](#性能验证)
6. [故障排除](#故障排除)
7. [进阶优化](#进阶优化)

## 系统要求

### 操作系统
- **macOS**: 12.x 或更高（Big Sur 或更新）
- **架构**: ARM64 (`aarch64`)
- **编译器**: Xcode 14 或更高（包含 Clang）

### 内存要求
- 至少 **4GB** RAM（用于编译大型扩展）

### CPU 特性
- NEON 支持（所有 ARM64 CPU 都支持）
- SVE 支持（仅部分新硬件，可选）

```bash
# 检查架构和编译器
uname -m              # 应该输出 arm64
clang --version        # 应该输出 Apple Clang 16 或更高
```

## 安装依赖

### 1. 安装 PostgreSQL

```bash
# 安装 PostgreSQL 16（推荐）
brew install postgresql@16

# 或安装其他版本（13, 14, 15）
# brew install postgresql@15
```

### 2. 安装 pgvector 源码

```bash
cd /path/to/pgvector

# 确保在 pgvector 目录下
cd /Users/a111/Desktop/code/pgvector
```

### 3. 设置环境变量

```bash
# 添加 PostgreSQL 到 PATH
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

# 设置 PostgreSQL 编译标志
export LDFLAGS="-L/opt/homebrew/opt/postgresql@16/lib"
export CPPFLAGS="-I/opt/homebrew/opt/postgresql@16/include"

# 验证 PostgreSQL 安装
/opt/homebrew/opt/postgresql@16/bin/pg_config --version
```

## 编译步骤

### 1. 清理旧的构建文件

```bash
cd /Users/a111/Desktop/code/pgvector

# 删除编译产物
rm -rf .clang_modules/ src/*.o

# 清理缓存
make clean 2>/dev/null || echo "Clean successful"
```

### 2. 设置编译标志（Mac ARM64 优化）

编辑 `Makefile`，添加 ARM64 特定的优化标志：

```makefile
# Mac ARM doesn't always support -march=native
ifeq ($(shell uname -s), Darwin)
	ifeq ($(shell uname -p), arm)
		# ARM64 优化 - 使用自动向量化
		OPTFLAGS = -march=armv8-a+crc+dotprod
	endif
endif

# 自动向量化标志
PG_CFLAGS += $(OPTFLAGS) -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math
```

**说明**：
- `march=armv8-a+crc+dotprod`: 启用 ARMv8.2 特性（CRC, 点积优化）
- `-ftree-vectorize`: 自动向量化
- `-fassociative-math`: 允许编译器重排浮点运算以提高性能
- `-fno-signed-zeros`: 忽略有符号零的优化
- `-fno-trapping-math`: 忽略浮点异常的优化

### 3. 编译

```bash
# 编译 pgvector（仅当前目录，不安装）
make

# 编译并安装
make

# 或者完整编译和安装
make install
```

**预期输出**：
```
clang ... src/bitutils.o
clang ... src/bitvec.o
...
clang ... src/neon_vector.o
clang ... src/neon_halfvec.o
...
```

### 4. 检查编译结果

```bash
# 检查是否生成了 NEON 对象文件
ls -lh src/*.o | grep neon

# 验证 NEON/SVE 支持（如果有）
otool -t src/neon_vector.o | grep -i neon
```

**预期结果**：
- 应该看到 `ARM64` 相关符号
- 如果有 SVE 支持，可以看到 SVE 指令

## 测试安装

### 1. 安装到 PostgreSQL

```bash
# 安装 pgvector 到 PostgreSQL
make install

# 验证安装
psql -c "SELECT version()" | grep vector
```

### 2. 测试基本功能

```sql
-- 创建测试表
CREATE TABLE test_vectors (
    id SERIAL PRIMARY KEY,
    v vector(1536),
    half halfvec(1536)
);

-- 插入测试数据
INSERT INTO test_vectors (v, halfvec) VALUES
    ('[1, 2, 3, 4]'::vector),
    ('[1, 2, 3, 4]'::halfvec);

-- 测试距离计算
SELECT
    v <-> halfvec AS halfvec_distance
FROM test_vectors;
```

### 3. 测试索引创建

```sql
-- 创建表并添加 HNSW 索引
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    v vector(1536)
);

CREATE INDEX ON items
USING hnsw
WITH (
    lists = 100,          -- HNSW 的列表数量
    m = 16,                 -- 连接数
    ef_construction = 64     -- 构建时的候选列表大小
);

-- 插入测试数据
INSERT INTO items (v)
SELECT * FROM generate_series(1, 1000);

-- 查询最近邻
SELECT id, v <-> $1 as distance
FROM items
ORDER BY distance
LIMIT 10;
```

## 性能验证

### 1. 检查 NEON 优化是否生效

```sql
-- 比较 NEON 优化 vs 默认实现
EXPLAIN ANALYZE
SELECT * FROM generate_series(1, 1000) as v
ORDER BY v <-> random() LIMIT 1;

-- 查看执行计划，确认使用了 NEON 向量化
```

**预期结果**：
- 执行计划中应该看到 `Vector Scan` 节点使用了 NEON 向量化

### 2. 性能基准测试

```sql
-- 性能测试脚本
EXPLAIN (SELECT v <-> random() FROM generate_series(1, 10000) LIMIT 1);

-- 批量距离计算
EXPLAIN (SELECT sum((v <-> random()) OVER ()) AS total_distance
FROM generate_series(1, 10000));

-- 与默认实现对比
```

**预期性能提升**：
- NEON 优化：**2-3x** 加速（128-bit 向量，4个 float 并行）
- SVE 优化（如果支持）：**4-6x** 加速（可变长度向量）

## 故障排除

### 问题 1: "pg_config: Command not found"

**原因**: PostgreSQL 未安装或不在 PATH 中

**解决方案**：
```bash
# 安装 PostgreSQL
brew install postgresql@16

# 设置环境变量
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

# 重新编译
make clean
make
make install
```

### 问题 2: 编译错误："unknown type name 'NEON_F32x4_t'"

**原因**: neon.h 中使用了错误的 NEON 内联函数名（x86 架构命名）

**解决方案**：
确保 `neon.h` 使用 ARM64 的 NEON 函数名（已修正）：
- `vld1q_f32` - 加载 128 位向量
- `vaddq_f32` - 向量加法
- `vsubq_f32` - 向量减法
- `vmlaq_f32` - 乘加（Multiply-Add）
- `vmulq_f32` - 向量乘法
- `vabsq_f32` - 绝对值

### 问题 3: 链接错误："Undefined symbols for architecture arm64"

**原因**: PostgreSQL 的 `arm_neon.h` 宏展中的 NEON 内联函数与 neon.h 中定义的不匹配

**解决方案**：
优先使用内置 NEON 函数，禁用 PostgreSQL 的 arm_neon.h 扩展：

```bash
# 添加编译标志以禁用 Postgres 内联函数
PG_CFLAGS += -DUSE_ARM_NEON_INTRINSICS

# 重新编译
make clean
make
```

### 问题 4: "vaddvq_f32" 未定义

**原因**: neon.h 中某些函数缺失定义

**解决方案**：
确保 `neon.h` 包含所有必要的函数定义（已修正）：
- `vaddvq_f32`, `vsubq_f32`, `vmlaq_f32`, `vmulq_f32`, `vabsq_f32`
- `ld1q_f32`, `vaddq_f32`, `vmlaq_f32`, `vmulq_f32`, `vabsq_f32`

## 进阶优化

### 1. 使用 Clang 内置 NEON 函数

修改 `neon.h`，直接使用 clang 的内置 NEON 函数：

```c
// 使用 clang 内置函数（最佳性能）
static inline float32x4_t
NeonAddvqF32(float32x4_t a, float32x4_t b)
{
	return vaddq_f32(a, b);
}

static inline float32x4_t
NeonMlaqF32(float32x4_t a, float32x4_t b, float32x4_t c)
{
	return vmlaq_f32(a, b, c);  // FMA 指令
}
```

### 2. 静态断言

```make
# 验证 NEON/SVE 代码是否被编译
nm src/*.o | grep -i "sve"

# 检查 NEON 指令的使用
objdump -t src/neon_vector.o | grep -i neon
```

### 3. 性能分析

```sql
-- 使用 EXPLAIN ANALYZE 查看执行计划
EXPLAIN ANALYZE
SELECT * FROM items
WHERE v <-> $1 ORDER BY v <-> $2 LIMIT 10;

-- 使用 pg_stat_statements 分析查询性能
SELECT * FROM pg_stat_statements
WHERE query LIKE '%v <-> %';
```

## 验证 NEON/SVE 支持

### 检测 CPU 特性

```bash
# 检查是否支持 NEON
sysctl -n machdep.cpu.cpu.brand
# 应该看到: Apple, Qualcomm, Marvell, etc.

# 检查 SVE（仅部分硬件支持）
sysctl -n machdep.cpu.brand
# 可能看到: Fujitsu A64FX（支持 SVE）

# 检查 NEON 扩展支持
sysctl -n machdep.cpu.features
# 应该看到: neon, crypto, sha3, aes, pmull, crc32
```

### 检查编译器优化

```bash
# 验证是否启用自动向量化
make clean
CFLAGS="-Wall -O2 -ftree-vectorize -ffast-math" make

# 检查目标指令
objdump -t src/vector.o | grep -i "vadd"
```

**预期结果**：
- 应该看到大量 NEON 指令：`vadd.f32`, `vsub.f32`, `vmla.f32`, `vfmul.f32`

## 编译成功标志总结

对于 **macOS ARM64** 环境，推荐以下编译标志组合：

```makefile
# Mac ARM64 优化
ifeq ($(shell uname -s), Darwin)
	ifeq ($(shell uname -p), arm)
		# ARMv8.2 优化
		OPTFLAGS = -march=armv8.2-a+crc+dotprod
	endif
endif

# 优化标志
PG_CFLAGS += $(OPTFLAGS) -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math
```

这些标志将确保：
1. 使用 NEON 自动向量化
2. 启用 CRC 和点积优化
3. 允许重排以提高性能
4. 忽略精度问题以提高速度
5. 生成最优化的 ARM64 代码

## 附录

### A. 常用的 ARM64 NEON 指令

| 指令 | 描述 | 硬件 |
|-----|------|------|
| `vadd.f32` | 向量加法 | NEON |
| `vsub.f32` | 向量减法 | NEON |
| `vmla.f32` | 乘加 | NEON |
| `vmla.f32` | 乘加 | NEON (FMA) |
| `vmul.f32` | 向量乘法 | NEON |
| `vabs.f32` | 绝对值 | NEON |
| `vld1.f32` | 加载向量 | NEON (128-bit) |
| `vst1.f32` | 存储向量 | NEON (128-bit) |
| `vld1q.f32` | 加载向量 | NEON (128-bit) |

### B. 性能提升预期

| 类型 | NEON 加速比 | 说明 |
|------|----------|------|
| Vector (FP32) | 2-3x | 128-bit SIMD，4 个 float 并行 |
| HalfVector (FP16) | 3-4x | 128-bit SIMD，8 个 half 并行，但需转换到 FP32 |
| SVE (FP32) | 4-6x | 可变长度 SIMD，性能最佳 |

### C. 相关资源

- [ARM NEON Intrinsics Reference](https://developer.arm.com/arch/is- arm/Reference/Arm-NEON_intrinsics/index.html)
- [PostgreSQL Source Code](https://github.com/postgres/postgres/tree/master/src/include/port/simd.h)
- [Clang Vector Extensions](https://clang.llvm.org/docs/Language/extending/clangLanguageExtensions.html#vector-extensions)

---

**编译检查清单**：

- [x] 使用 `vld1q_f32`（加载向量）
- [x] 使用 `vaddq_f32`, `vsubq_f32`（向量加减）
- [x] 使用 `vmlaq_f32`（FMA 指令）
- [x] 使用 `vabsq_f32`（取绝对值）
- [x] 使用 `vld1q_f32`（水平求和）

编译成功后，NEON 优化将自动激活，显著提升 ARM64 平台上的 pgvector 性能。