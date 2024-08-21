python -u main.py run_all --forecast_model GRU         -num_head 8 --tau 10 --lr_da 0.01  --market csi300 --data_dir cn_data --rank_label False --begin_valid_epoch 30 --alpha 158 > logs/gru300_158_da.log 2>&1
python -u main.py run_all --forecast_model LSTM        -num_head 8 --tau 10 --lr_da 0.01  --market csi300 --data_dir cn_data --rank_label False --begin_valid_epoch 30 --alpha 158 > logs/lstm300_158_da.log 2>&1
python -u main.py run_all --forecast_model ALSTM       -num_head 8 --tau 10 --lr_da 0.01  --market csi300 --data_dir cn_data --rank_label False --begin_valid_epoch 30 --alpha 158 > logs/alstm300_158_da.log 2>&1
python -u main.py run_all --forecast_model Transformer -num_head 8 --tau 10 --lr_da 0.01  --market csi300 --data_dir cn_data --rank_label False --begin_valid_epoch 30 --alpha 158 > logs/tfm300_158_da.log 2>&1
python -u main.py run_all --forecast_model GRU         -num_head 8 --tau 10 --lr_da 0.01  --market csi300 --data_dir cn_data --rank_label False --begin_valid_epoch 30 --alpha 360 > logs/gru300_158_da.log 2>&1
python -u main.py run_all --forecast_model LSTM        -num_head 8 --tau 10 --lr_da 0.01  --market csi300 --data_dir cn_data --rank_label False --begin_valid_epoch 30 --alpha 360 > logs/lstm300_158_da.log 2>&1
python -u main.py run_all --forecast_model ALSTM       -num_head 8 --tau 10 --lr_da 0.01  --market csi300 --data_dir cn_data --rank_label False --begin_valid_epoch 30 --alpha 360 > logs/alstm300_158_da.log 2>&1
python -u main.py run_all --forecast_model Transformer -num_head 8 --tau 10 --lr_da 0.01  --market csi300 --data_dir cn_data --rank_label False --begin_valid_epoch 30 --alpha 360 > logs/tfm300_158_da.log 2>&1

