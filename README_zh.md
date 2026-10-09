# BatchMatmulMaxSum：官方模板正确性基线

## 这版是什么
保持官方 `run_kernel` 签名不变，补齐了Host元数据解析和Device计算。
不再使用前面的PyTorch扩展；只需CANN、C++工具链、numpy、ml_dtypes。
支持FP16/BF16、四种存储布局、非对齐M/N、K为8倍数、全负相似度。
用输入16位实际存储值解码成FP32，先完成K点积，再Max(N)，最后Sum(M)。
K与M使用补偿求和，Max由首个有效列初始化。

**这是单Vector核上的标量正确性基线，不是Cube/Vector融合性能版。**
大shape可能超时，不能声称15个官方case全部通过，更不能作为高分版本。
当前制作环境没有CANN/NPU：CPU逻辑检查不代表Ascend C编译或上板通过。
FP32补偿求和也不保证任意极端输入都满足FP64 golden误差要求。

## 文件分工
- kernel.asc：bmms::Parse为Host解析；Decode/Accumulate与全局函数为Device计算；run_kernel为Host启动。
- main.asc：本地ACL内存、文件读写、启动和同步；默认仍运行官方case0，可用参数运行额外用例。
- CMakeLists.txt：沿用模板。
- run.sh：沿用模板流程，增加NPU_ARCH环境变量传递。
- scripts/BatchMatmulMaxSum.py、gen_data.py、verify_result.py：原模板参考与单case验证，未修改。
- scripts/logic_check.py：CPU shim编译实际算法，验证数学逻辑和元数据，绝非NPU模拟器。
- scripts/test_npu.py：96个额外小规模上板正确性测试。

## 在WSL或Linux运行
Windows的 D:\CANN 对应WSL的 /mnt/d/CANN。
将zip解压到该目录后：

```bash
cd /mnt/d/CANN/BatchMatmulMaxSum_baseline
python3 -m pip install numpy ml_dtypes
source /你的CANN安装目录/set_env.sh
bash run.sh
python3 scripts/test_npu.py
```

如果解压工具没有生成外层目录，直接进入含run.sh的目录。
若环境脚本没有设置ASCEND_HOME_PATH，手动指定为包含set_env.sh的目录：

```bash
export ASCEND_HOME_PATH=/你的CANN安装目录
bash run.sh
```

模板默认架构dav-2201。其他设备必须用实际匹配的架构：

```bash
NPU_ARCH=实际目标架构 bash run.sh
```

普通WSL没有NPU驱动和硬件时，不能直接运行ACL上板程序。
只验证CPU逻辑可用：

```bash
sudo apt install g++
python3 scripts/logic_check.py
```

## main参数
```text
./batch_matmul_max_sum_custom B M N K dtype tx1 tx2
```
dtype为1(FP16)或2(BF16)，tx为0或1。input/x1.bin、input/x2.bin必须
由对应dtype/storage shape的实际数据生成；修改命令参数不会自动修改输入文件。

## 三人接下来的协作
你：掌握Parse、main内存管理、run_kernel；再设计Tiling与任务/工作区协议。
成员A：用03.03的Matmul API替换标量三重循环，关闭Bias，正确设置布局；
随后按03.05连接Vector Max/Sum，避免完整相似度矩阵落GM。
成员B：维护golden和上板用例，再做大K、精度和msprof测试。
性能改造时保持run_kernel ABI和原模板参考文件不变。

## 如何读代码
1. Parse：从shape指针与属性恢复B/M/N/K，并检查输出与输入类型。
2. Decode：把FP16/BF16的原始16位存储转为FP32。
3. Device循环：batch -> query行 -> document列 -> embedding维。
4. n==0分支：全负相似度也能正确选最大值。
5. run_kernel：启动一个Vector核；不使用PyTorch或Tensor.size()。
