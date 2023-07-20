# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# $1: TP size
# $2: PP size

export NVIDIA_TF32_OVERRIDE=0
tensor_para_size=$1
pipeline_para_size=$2
int8_mode=$3
total_gpu_count=$(echo "scale=2; ${tensor_para_size} * ${pipeline_para_size} " | bc)

# config for bloom
vocab_size=50272
start_id=2
end_id=2
model_variant=opt-pre
request_batch_size_list=(1 8 16)
input_length_list=(2048 4096)
request_output_len_list=(100)

if [ -z "${int8_mode}" ]; then
  int8_mode=0
fi

if [ "${int8_mode}" = 0 ]; then
  logdir="opt-TP${tensor_para_size}-PP${pipeline_para_size}-fp16-log"
else
  logdir="opt-TP${tensor_para_size}-PP${pipeline_para_size}-int8-log"
fi
if [ ! -f "${logdir}" ]; then
  mkdir "${logdir}" -p
fi

all_log="${logdir}/all-log.log"

echo -e "| model size | Batch Size | Input length | Output length | Decode value | Precision | FT latency (ms) |" > "$all_log"
echo -e "|:----------:|:----------:|:------------:|:-------------:|:------------:|:---------:|:---------------:|" >> "$all_log"

cat /proc/cpuinfo >"${logdir}"/cpuinfo.txt
nvidia-smi >"${logdir}/gpuinfo.txt"

model_size_list=("175b")
for model_size in "${model_size_list[@]}"; do
  if [ "$model_size" = "6.7b" ]; then
    head_num=32
    size_per_head=128
    inter_size=$(echo "scale=2; $head_num * ${size_per_head} * 4 " | bc)
    num_layer=32
  elif [ "$model_size" = "175b" ]; then
    head_num=56
    size_per_head=128
    inter_size=$(echo "scale=2; $head_num * ${size_per_head} * 4 " | bc)
    num_layer=280
  else
    echo "model_size error"
    exit 1
  fi

  if [ "${int8_mode}" = 0 ]; then
    model_dir="../models/huggingface-models/c-model/opt-${model_size}/${total_gpu_count}-gpu/"
    print_log='{printf "| %5s | %3d | %4d | %4d | %10s %5s | FP16 | %7.2f |\n", model_size, batch_size, input_length, request_output_len,
              decode_type, decode_value, ft_latency}'
  else
    model_dir="../models/huggingface-models/c-model/opt-${model_size}-int8/${total_gpu_count}-gpu/"
    print_log='{printf "| %5s | %3d | %4d | %4d | %10s %5s | INT8 | %7.2f |\n", model_size, batch_size, input_length, request_output_len,
              decode_type, decode_value, ft_latency}'
  fi

  # decode_type_list=(greedy-search beam-search sampling)
  decode_type_list=(greedy-search)

  for decode_type in "${decode_type_list[@]}"; do
    if [ "$decode_type" = "greedy-search" ]; then
      decode_values=(1)
    elif [ "$decode_type" = "beam-search" ]; then
      decode_values=(4)
    elif [ "$decode_type" = "sampling" ]; then
      decode_values=(0.9)
    else
      echo "decode_type error"
      exit 1
    fi

    for request_batch_size in "${request_batch_size_list[@]}"; do
      for input_length in "${input_length_list[@]}"; do
        for request_output_len in "${request_output_len_list[@]}"; do
          for decode_value in "${decode_values[@]}"; do
            if [ "$decode_type" = "greedy-search" ]; then
              beam_width=$decode_value
              topk=1
              topp=0.0
            elif [ "$decode_type" = "beam-search" ]; then
              beam_width=$decode_value
              topk=0
              topp=0.0
            elif [ "$decode_type" = "sampling" ]; then
              beam_width=1
              if [[ $decode_value == +([[:digit:]]) ]]; then
                topk=$decode_value
                topp=0.6
              else
                topk=0
                topp=$decode_value
              fi
            fi

            tmp_log=${logdir}/batchsize-${request_batch_size}-decode_value-${decode_value}-${input_length}-${request_output_len}-${decode_type}-${decode_value}.log

            python ../examples/pytorch/gpt/utils/generate_start_ids.py --max_batch_size "${request_batch_size}" --max_input_length "${input_length}"
            ./bin/gpt_gemm "${request_batch_size}" "${beam_width}" "${input_length}" "${head_num}" "${size_per_head}" "${inter_size}" ${vocab_size} 1 "${tensor_para_size}"

            python ../examples/pytorch/gpt/utils/generate_gpt_config.py \
              --max_seq_len 1024 \
              --beam_width ${beam_width} \
              --head_num ${head_num} \
              --size_per_head ${size_per_head} \
              --inter_size "${inter_size}" \
              --num_layer ${num_layer} \
              -v "${vocab_size}" \
              -d fp16 \
              -topk "${topk}" \
              -topp "${topp}" \
              --tensor_para_size "${tensor_para_size}" \
              --pipeline_para_size "${pipeline_para_size}" \
              --model_dir "${model_dir}" \
              --request_batch_size "${request_batch_size}" \
              --request_output_len "${request_output_len}" \
              --start_id "${start_id}" \
              --end_id "${end_id}" \
              --model_variant "${model_variant}" \
              --int8_mode "${int8_mode}"
            mpirun -n "${total_gpu_count}" --allow-run-as-root ./bin/multi_gpu_gpt_example .tmp.config.ini 2>&1 | tee "${tmp_log}"
            ft_latency=$(tail -n 1 "${tmp_log}" | head -n 1 | awk '{print $17}')
            echo "" | awk -v ft_latency="$ft_latency" \
              -v batch_size="$request_batch_size" \
              -v input_length="${input_length}" -v request_output_len="$request_output_len" \
              -v model_size="${model_size}" -v decode_value="$decode_value" -v decode_type="$decode_type" \
              "${print_log}" >>"$all_log"

            rm .tmp.config.ini

          done # decode_values
        done   # request_output_len
      done     # input_length
    done       # batch_size
  done         # decode_type
done           # model_size
