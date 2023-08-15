#!/usr/bin/env bash

set -eu

# comment to run on CPUs, uncomment for GPUs
cudnn_path="/project/d121/ruestefa/spack/daint/cudnn-8.1.1.33-11.2/nvhpc-21.3/gofqrtudwdpdyuy23ettnsjahdrrnmub"
export LD_LIBRARY_PATH="${cudnn_path}/lib64:$LD_LIBRARY_PATH"

# Optional argument: Path to either the model parent directory, in which case
# a default model is used, or to a specific model directory
model_dir="${1:-"/scratch/snx3000/gbertoli/icon/regress-radiation/saved_models"}"
# default_model="patch_shrNhbrWeightslvl0_all_heights_BN_mse_datav2IG_SSD_MT_featNormv2_fit_seq_MergedLwupLwdnSwupSwdn/serialized_model_steps_182400_238640_80560_188480"  # 60 levels
default_model="patch_shrNhbrWeightslvl0_all_heights_BN_mse_energyconsv4W6e-1_lwsw_datav8IG_featNormv2_fit_flatind_bs1024_unetv1/serialized_model_step_800421"  # 70 levels
[[ -f "${model_dir}/saved_model.pb" ]] || model_dir+="/${default_model}"

eng_type=tf_c

# run inference
cmd=(./multiple_fields_model.exe "${model_dir}" "${eng_type}")
echo "${cmd[@]}"
srun "${cmd[@]}"
