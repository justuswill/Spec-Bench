# M5 benchmark

# AR v1.5     MAT: 1.0
python-spec -m evaluation.inference_baseline --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5-vanilla-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-spec -m evaluation.inference_baseline --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5-vanilla-float16-temp-0.7-m5-2 --bench-name spec_bench --temperature 0.7 --dtype float16
python-spec -m evaluation.inference_baseline --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5-vanilla-float16-temp-0.7-m5-3 --bench-name spec_bench --temperature 0.7 --dtype float16
# AR v1.3     MAT: 1.0
python-specl -m evaluation.inference_baseline --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-vanilla-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_baseline --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-vanilla-float16-temp-0.7-m5-2 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_baseline --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-vanilla-float16-temp-0.7-m5-3 --bench-name spec_bench --temperature 0.7 --dtype float16
# SPS         MAT: 2.0458, 2.0171, 2.0001
python-specl -m evaluation.inference_sps --drafter-path double7/vicuna-68m --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-sps-68m-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_sps --drafter-path double7/vicuna-68m --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-sps-68m-float16-temp-0.7-m5-2 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_sps --drafter-path double7/vicuna-68m --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-sps-68m-float16-temp-0.7-m5-3 --bench-name spec_bench --temperature 0.7 --dtype float16
# MEDUSA      MAT: 2.4946
python-specl -m evaluation.inference_medusa --model-path FasterDecoding/medusa-vicuna-7b-v1.3 --base-model lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-medusa-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_medusa --model-path FasterDecoding/medusa-vicuna-7b-v1.3 --base-model lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-medusa-float16-temp-0.7-m5-2 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_medusa --model-path FasterDecoding/medusa-vicuna-7b-v1.3 --base-model lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-medusa-float16-temp-0.7-m5-3 --bench-name spec_bench --temperature 0.7 --dtype float16
# EAGLE       MAT: 3.3815(4), 3.4350, 3,4232
python-specl -m evaluation.inference_eagle --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3 --base-model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-eagle-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_eagle --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3 --base-model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-eagle-float16-temp-0.7-m5-2 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_eagle --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3 --base-model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-eagle-float16-temp-0.7-m5-3 --bench-name spec_bench --temperature 0.7 --dtype float16
# EAGLE 2     MAT: 4.1350, 4.1015, 4.1517
python-specl -m evaluation.inference_eagle2 --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3 --base-model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-eagle2-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_eagle2 --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3 --base-model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-eagle2-float16-temp-0.7-m5-2 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_eagle2 --ea-model-path yuhuili/EAGLE-Vicuna-7B-v1.3 --base-model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-eagle2-float16-temp-0.7-m5-3 --bench-name spec_bench --temperature 0.7 --dtype float16
# LOOKAHEAD   MAT: 1.6394
USE_LADE=1 python-specl -m evaluation.inference_lookahead --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-lookahead-float16-temp-0.7-m5-1 --level 5 --window 7 --guess 7 --bench-name spec_bench --dtype float16
USE_LADE=1 python-specl -m evaluation.inference_lookahead --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-lookahead-float16-temp-0.7-m5-2 --level 5 --window 7 --guess 7 --bench-name spec_bench --dtype float16
USE_LADE=1 python-specl -m evaluation.inference_lookahead --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-lookahead-float16-temp-0.7-m5-3 --level 5 --window 7 --guess 7 --bench-name spec_bench --dtype float16
# PLD         MAT: 1.7255
python-specl -m evaluation.inference_pld --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-pld-float16-m5-1 --bench-name spec_bench --dtype float16
python-specl -m evaluation.inference_pld --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-pld-float16-m5-2 --bench-name spec_bench --dtype float16
python-specl -m evaluation.inference_pld --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-pld-float16-m5-3 --bench-name spec_bench --dtype float16
# HYDRA       MAT: 3.5637
python-specl -m evaluation.inference_hydra --model-path ankner/hydra-vicuna-7b-v1.3 --base-model lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-hydra-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_hydra --model-path ankner/hydra-vicuna-7b-v1.3 --base-model lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-hydra-float16-temp-0.7-m5-2 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_hydra --model-path ankner/hydra-vicuna-7b-v1.3 --base-model lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-hydra-float16-temp-0.7-m5-3 --bench-name spec_bench --temperature 0.7 --dtype float16
# RECYCLING   MAT: 2.7281, 2.7278, 2.7389
python-specl -m evaluation.inference_recycling --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-recycling-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_recycling --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-recycling-float16-temp-0.7-m5-2 --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_recycling --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-recycling-float16-temp-0.7-m5-3 --bench-name spec_bench --temperature 0.7 --dtype float16
# SAMD        MAT: 4.2585, 4.0982, 4.2172
python-specl -m evaluation.inference_samd --tree_model_path /extra/ucibdl1/jcwill/specbench/models/EAGLE-Vicuna-7B-v1.3 --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-samd-float16-temp-0.7-m5-1 --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle2 --attn_implementation sdpa --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_samd --tree_model_path /extra/ucibdl1/jcwill/specbench/models/EAGLE-Vicuna-7B-v1.3 --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-samd-float16-temp-0.7-m5-2 --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle2 --attn_implementation sdpa --bench-name spec_bench --temperature 0.7 --dtype float16
python-specl -m evaluation.inference_samd --tree_model_path /extra/ucibdl1/jcwill/specbench/models/EAGLE-Vicuna-7B-v1.3 --model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-samd-float16-temp-0.7-m5-3 --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --tree_method eagle2 --attn_implementation sdpa --bench-name spec_bench --temperature 0.7 --dtype float16
# PTP v1.3    MAT:
python-spec -m evaluation.inference_ptp --student-path /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna-ultra --base-model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-ptp-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-spec -m evaluation.inference_ptp --student-path /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna-ultra --base-model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-ptp-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-spec -m evaluation.inference_ptp --student-path /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna-ultra --base-model-path lmsys/vicuna-7b-v1.3 --model-id vicuna-7b-v1.3-ptp-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
# PTP v1.5    MAT:
python-spec -m evaluation.inference_ptp --student-path /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna-ultra --model-id vicuna-7b-v1.5-ptp-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-spec -m evaluation.inference_ptp --student-path /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna-ultra --model-id vicuna-7b-v1.5-ptp-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
python-spec -m evaluation.inference_ptp --student-path /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna-ultra --model-id vicuna-7b-v1.5-ptp-float16-temp-0.7-m5-1 --bench-name spec_bench --temperature 0.7 --dtype float16
