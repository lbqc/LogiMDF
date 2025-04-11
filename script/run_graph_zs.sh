export CUDA_VISIBLE_DEVICES=2
lrs=("1e-5")
bert_lrs=("5e-6")

# model_names=("roberta_fol_text")
model_names=("bert_fol_hgraph")
train_scales=("0.1" "0.2" "0.3" "0.4" "0.5" "1")
# model_names=("lstm_fol")
input_types=("tt")
# datasets=("dt-" "fm-" "hc-" "la-" "vast_10" "vast" "fm-la" "la-fm" "hc-dt" "dt-hc" "dt" "fm" "hc" "la") # "vast" "vast_10" "vast" "fm-la" "la-fm" "hc-dt" "dt-hc" "dt" "fm" "hc" "la"
# datasets=("vast_10" "vast" "dt" "fm" "hc" "la")
# datasets=('vast')
# datasets=("dt" "fm" "hc" "la" "vast_10")
# datasets=("fm-la" "la-fm" "hc-dt" "dt-hc")
datasets=("dt-" "fm-" "hc-" "la-")
# datasets=("vast")
# datasets=("hc-" "dt-")
llm_names=("3llms")


seeds=("0" "1" "2")
# seeds=("1" "2" "3")
# seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "13" "17" "19" "23")
# seeds=("3")
# seeds=("0")

dropouts=("0.2")


label_ratios=(1)
FAD_ratios=(1)
RAD_ratios=(0)
for lr in ${lrs[*]}
do
    for bert_lr in ${bert_lrs[*]}
    do
        for model_name in ${model_names[*]}
        do
            for dataset in ${datasets[*]}
            do
                for seed in ${seeds[*]}
                do
                    for dropout in ${dropouts[*]}
                    do
                        for llm_name in ${llm_names[*]}
                        do
                            for train_scale in ${train_scales[*]}
                            do
                                            python3 ../codes/train.py \
                                            --lr $lr \
                                            --model_name $model_name \
                                            --dataset $dataset \
                                            --seed $seed \
                                            --dropout $dropout \
                                            --valset_ratio 0.0 \
                                            --bert_lr $bert_lr \
                                            --num_epoch 20 \
                                            --max_seq_len 60 \
                                            --nodes_num 80 \
                                            --edge_num 400 \
                                            --gcn_num 2 \
                                            --train_scale $train_scale \
                                            --llm_name $llm_name \
                                            --parent_need 1 \
                                            # --semantic_similarity_threshold -1 \
                                            # --child2parent_need 1 \
                                            # --hyperedges_num 4 \
                                            # --with_text 1
                                            # --parent_need 0 \
                                            # --with_text 0 \
                                            # --use_prompt 0 \
                                            # --batch_size 64 \
                                            #  > "logs/"$learning_rate"_"$max_epoch"_"$batch_size".log" 
                            done
                        done
                    done
                done
            done
        done
    done
done

# nohup bash script/PStance_bernie_run.sh > logs/_PStance_bernie.out &
