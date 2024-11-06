`ddp_training.sh` is the training script for distributed multi-hop tracing fine-tuning. We uploaded 2k SFT training data, `Hotpot_train_data.json`, to help understand the dataset format. 

The training relies on the **ms-swift** framework ([https://github.com/modelscope/ms-swift](https://github.com/modelscope/ms-swift)) and uses **flash-attn** acceleration ([https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)).
