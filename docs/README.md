# 基于`FasterTransformer`的部署方案
## 编译FT源码
```bash
git clone https://github.com/NVIDIA/FasterTransformer.git
mkdir -p FasterTransformer/build
cd FasterTransformer/build
git submodule update --init --recursive
cmake -DSM=70,75,80,86,89,90 -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON -DBUILD_PYT=ON ..
make -j12
```

> **Note**
> - 编译使用的环境镜像为Pytorch的ngc镜像: `nvcr.io/nvidia/pytorch:22.12-py3`
> - 不使用`-DSM=xx`编译默认设置为`70,75,80,86`


## Bloom系列  
以`phoenix-inst-chat-7b-v1.1`为例  

### 模型转换(hf-->ft)  
```bash
export MODEL=phoenix-inst-chat-7b-v1.1
python3 ../examples/pytorch/gpt/utils/huggingface_bloom_convert.py -i "../models/$MODEL" -o "../c-model/$MODEL" -tp 4 -dt fp16 -p 1 -v
```

### 运行Benchmark测试
```bash
bash ../benchmarks/gpt/bloom_benchmark.sh 4 1 0
```

| 参数   | 描述                                                             |
|:-----|:---------------------------------------------------------------|
| 参数 1 | Tensor 并行数                                                     |
| 参数 2 | Pipeline 并行数                                                   |
| 参数 3 | 0: `fp16`<br> 1: `weight-only int8`<br>  2: `SmoothQuant int8` |

> **Note**
> - weight-only int8: 仅权重为`Int8`, 无实际加速效果只节省显存
> - SmoothQuant int8: w8a8-int8推理, 有实际加速效果约11%
> - build目录下生成的`gemm_config.in`配置文件用于优化矩阵乘法运算，Tensorrt Build时有类似功能
> - benchmark脚本生成的`.tmp.config.ini`可用于`tritonserver`部署

## Triton Inference Server部署

# 基于`vLLM`的部署方案




