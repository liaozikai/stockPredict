# # main.py
import argparse
import torch
import numpy as np
import random
from exp.exp_model import Exp_Model
import os
import pandas as pd
import time
import csv

fix_seed = 100
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='generating')

# Load data
parser.add_argument('--root_path', type=str, default='./data/2016', help='root path of the data files')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--sequence_length', type=int, default=10, help='length of input sequence')
parser.add_argument('--prediction_length', type=int, default=None, help='prediction sequence length')
parser.add_argument('--target_dim', type=int, default=1, help='dimension of target')
parser.add_argument('--input_dim', type=int, default=6, help='dimension of input')
parser.add_argument('--hidden_size', type=int, default=128, help='encoder dimension')
parser.add_argument('--embedding_dimension', type=int, default=64, help='feature embedding dimension')

# Diffusion process
parser.add_argument('--diff_steps', type=int, default=1000, help='number of the diff step')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout')
parser.add_argument('--beta_schedule', type=str, default='linear', help='the schedule of beta')
parser.add_argument('--beta_start', type=float, default=0.0, help='start of the beta')
parser.add_argument('--beta_end', type=float, default=1.0, help='end of the beta')
parser.add_argument('--scale', type=float, default=0.1, help='adjust diffusion scale')

# Bidirectional VAE
parser.add_argument('--arch_instance', type=str, default='res_mbconv', help='path to the architecture instance')
parser.add_argument('--mult', type=float, default=1, help='mult of channels')
parser.add_argument('--num_layers', type=int, default=2, help='num of RNN layers')
parser.add_argument('--num_channels_enc', type=int, default=32, help='number of channels in encoder')
parser.add_argument('--channel_mult', type=int, default=2, help='number of channels in encoder')
parser.add_argument('--num_preprocess_blocks', type=int, default=1, help='number of preprocessing blocks')
parser.add_argument('--num_preprocess_cells', type=int, default=3, help='number of cells per block')
parser.add_argument('--groups_per_scale', type=int, default=2, help='number of cells per block')
parser.add_argument('--num_postprocess_blocks', type=int, default=1, help='number of postprocessing blocks')
parser.add_argument('--num_postprocess_cells', type=int, default=2, help='number of cells per block')
parser.add_argument('--num_channels_dec', type=int, default=32, help='number of channels in decoder')
parser.add_argument('--num_latent_per_group', type=int, default=8, help='number of channels in latent variables per group')

# Training settings
parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
parser.add_argument('--itr', type=int, default=5, help='experiment times')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=200, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0000, help='weight decay')
parser.add_argument('--zeta', type=float, default=0.5, help='trade off parameter zeta')
parser.add_argument('--eta', type=float, default=1.0, help='trade off parameter eta')

# Device
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')


def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# def main(args):
#     args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
#
#     if args.prediction_length is None:
#         args.prediction_length = args.sequence_length
#
#     print('Args in experiment:')
#     print(args)
#
#     Exp = Exp_Model
#     calModel = True
#
#     results = pd.DataFrame(columns=['Ticker', 'MAE', 'MSE','IC','ICIR','RANK IC','RANK ICIR','TRAIN_TIME'])
#     train_setting = 'tp{}_sl{}'.format(args.root_path.split(os.sep)[-1], args.sequence_length)
#
#     for idx, file in enumerate(os.listdir(args.root_path)):  # Iterate through all tickers
#         print('\\n\\nRunning on file {} ({}/{})...'.format(file, idx + 1, len(os.listdir(args.root_path))))
#         args.data_path = file
#         ticker = os.path.splitext(file)[0]
#         all_mae = []
#         all_mse = []
#         all_ic = []
#         all_icir = []
#         all_rank_ic = []
#         all_rank_icir = []
#
#         all_train_time = []
#
#         for ii in range(0, args.itr):
#             setting = args.data_path + '_' + train_setting
#             exp = Exp(args)  # single experiment
#
#             if calModel:
#                 # 计算每个模型的参数量
#                 gen_net_size = calculate_model_size(exp.gen_net)
#                 denoise_net_size = calculate_model_size(exp.denoise_net)
#                 pred_net_size = calculate_model_size(exp.pred_net)
#                 embedding_size = calculate_model_size(exp.embedding)
#
#                 # 计算总参数量
#                 total_model_size = gen_net_size + denoise_net_size + pred_net_size + embedding_size
#
#                 # 将字节数转换为兆字节（MB）
#                 total_megabytes = total_model_size / 2 ** 20
#
#                 print(f"Total Model Size(MB): {total_megabytes:.4f} MB")
#                 calModel = False
#
#             print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
#             try:
#                 start_time = time.time()
#                 exp.train(setting)
#                 end_time = time.time()
#                 all_train_time.append(end_time - start_time)
#             except Exception as e:
#                 print("An error occurred during training: {}".format(e))
#                 # Handle the exception as needed, e.g., set default values or continue to the next iteration
#                 # mae, mse, ic, icir, rank_ic, rank_icir = (None, None, None, None, None, None)
#                 continue  # Skip the rest of the loop and continue with the next iteration
#
#             print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
#             try:
#                 mae, mse, ic, icir, rank_ic, rank_icir = exp.test(setting)
#             except Exception as e:
#                 print("An error occurred during testing: {}".format(e))
#                 # Handle the exception as needed, e.g., set default values or continue to the next iteration
#                 # mae, mse, ic, icir, rank_ic, rank_icir = (None, None, None, None, None, None)
#                 continue  # Skip the rest of the loop and continue with the next iteration
#             all_mae.append(mae)
#             all_mse.append(mse)
#             all_ic.append(ic)
#             all_icir.append(icir)
#             all_rank_ic.append(rank_ic)
#             all_rank_icir.append(rank_icir)
#             torch.cuda.empty_cache()
#
#         print("================all_train_time========================")
#         print(all_train_time)
#         results = results.append({'Ticker': ticker, 'MAE': np.mean(np.array(all_mae)),
#                                   'MSE': np.mean(np.array(all_mse)),
#                                   'IC': np.mean(np.array(all_ic)),
#                                   'ICIR': np.mean(np.array(all_icir)),
#                                   'RANK_IC': np.mean(np.array(all_rank_ic)),
#                                   'RANK_ICIR': np.mean(np.array(all_rank_icir)),
#                                   'TRAIN_TIME': np.mean(np.array(all_train_time))
#                                   },
#                                  ignore_index=True)
#
#     folder_path = './results/'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     results.to_csv(folder_path + train_setting + '.csv', index=False)
#     print(results)
#



def main(args):
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.prediction_length is None:
        args.prediction_length = args.sequence_length
    print('Args in experiment:')
    print(args)

    Exp = Exp_Model
    calModel = True

    results = pd.DataFrame(columns=['Ticker', 'MAE', 'MSE', 'IC', 'ICIR', 'RANK_IC', 'RANK_ICIR', 'TRAIN_TIME'])
    train_setting = 'tp{}_sl{}'.format(args.root_path.split(os.sep)[-1], args.sequence_length)
    folder_path = './results/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(folder_path + train_setting + '.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.columns)
        writer.writeheader()

        for idx, file in enumerate(os.listdir(args.root_path)):  # Iterate through all tickers
            print('Running on file {} ({}/{})...'.format(file, idx + 1, len(os.listdir(args.root_path))))
            args.data_path = file
            ticker = os.path.splitext(file)[0]
            all_mae = []
            all_mse = []
            all_ic = []
            all_icir = []
            all_rank_ic = []
            all_rank_icir = []
            all_train_time = []

            for ii in range(0, args.itr):
                setting = args.data_path + '_' + train_setting
                exp = Exp(args)  # single experiment

                if calModel:
                    gen_net_size = calculate_model_size(exp.gen_net)
                    denoise_net_size = calculate_model_size(exp.denoise_net)
                    pred_net_size = calculate_model_size(exp.pred_net)
                    embedding_size = calculate_model_size(exp.embedding)
                    total_model_size = gen_net_size + denoise_net_size + pred_net_size + embedding_size
                    total_megabytes = total_model_size / 2 ** 20
                    print(f"Total Model Size(MB): {total_megabytes:.4f} MB")
                    calModel = False

                print('start training: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                try:
                    start_time = time.time()
                    exp.train(setting)
                    end_time = time.time()
                    all_train_time.append(end_time - start_time)
                except Exception as e:
                    print("An error occurred during training: {}".format(e))
                    continue

                print('start testing: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                try:
                    mae, mse, ic, icir, rank_ic, rank_icir = exp.test(setting)
                except Exception as e:
                    print("An error occurred during testing: {}".format(e))
                    continue
                all_mae.append(mae)
                all_mse.append(mse)
                all_ic.append(ic)
                all_icir.append(icir)
                all_rank_ic.append(rank_ic)
                all_rank_icir.append(rank_icir)
                torch.cuda.empty_cache()

            row = {'Ticker': ticker, 'MAE': np.mean(all_mae), 'MSE': np.mean(all_mse), 'IC': np.mean(all_ic), 'ICIR': np.mean(all_icir), 'RANK_IC': np.mean(all_rank_ic), 'RANK_ICIR': np.mean(all_rank_icir), 'TRAIN_TIME': np.mean(all_train_time)}
            print("-================================row=================")
            print(row)
            writer.writerow(row)
            csvfile.flush()  # 确保数据被写入文件

            # 在这里可以添加代码来下载文件，例如使用webbrowser模块
            # import webbrowser
            # webbrowser.open(folder_path + train_setting + '.csv')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)