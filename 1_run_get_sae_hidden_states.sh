python 1_get_feature_activation.py \
    --data_path /home/projects/SAE/sae_v0/data-llama3-8b/wikitext-103\
    --model_path /data2/pretrained_models/Llama-3-8B \
    --output_dir output/hidden_states_llama3_8b \
    --model_name llama3-8b \
    --tgt_layer 15 \
    --gpu_id 0 \

python 1_get_feature_activation.py \
    --data_path /home/projects/SAE/sae_v0/data-llama3-8b/2024_wiki\
    --model_path /data2/pretrained_models/Llama-3-8B \
    --output_dir output/hidden_states_llama3_8b \
    --model_name llama3-8b \
    --tgt_layer 15 \
    --gpu_id 0 \
