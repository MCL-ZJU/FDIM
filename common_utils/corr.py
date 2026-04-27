import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from common_utils import linear


def calc_content_correlation(results_df, pred_name, label_name):
    single_results = []
    group = results_df.groupby('ref_name')
    plcc_all, srocc_all, krocc_all, rmse_all = [], [], [], []
    for ref_name, seqs in group:
        x_tmp = list(seqs[pred_name])
        y_tmp = list(seqs[label_name])
        coef = linear.fit(x_tmp, y_tmp)
        curve = lambda x_tmp: linear.func_linear(x_tmp, coef)
        y_hat = curve(x_tmp)

        plcc = pearsonr(y_hat, y_tmp)[0]
        srocc = spearmanr(y_hat, y_tmp)[0]
        krocc, _ = kendalltau(y_hat, y_tmp)
        rmse = np.sqrt(np.mean((np.array(y_hat) - np.array(y_tmp)) ** 2))
        plcc_all.append(plcc)
        srocc_all.append(srocc)
        krocc_all.append(krocc)
        rmse_all.append(rmse)

        single_results.append({
            'SRC_name': ref_name,
            'PLCC': plcc,
            'SROCC': srocc,
            'KROCC': krocc,
            'RMSE': rmse,
        })

    return plcc_all, srocc_all, krocc_all, rmse_all, single_results


# def calc_dataset_correlation(results_df, pred_name, label_name):
#     x_tmp = list(results_df[pred_name])
#     y_tmp = list(results_df[label_name])
#     coef = linear.fit(x_tmp, y_tmp)
#     curve = lambda x_tmp: linear.func_linear(x_tmp, coef)
#     y_hat = curve(x_tmp)
#
#     plcc = pearsonr(y_hat, y_tmp)[0]
#     srocc = spearmanr(y_hat, y_tmp)[0]
#     krocc, _ = kendalltau(y_hat, y_tmp)
#     rmse = np.sqrt(np.mean((np.array(y_hat) - np.array(y_tmp)) ** 2))
#
#     return plcc, srocc, krocc, rmse

def calc_correlation(results_df, save_dir, save_name, metric, pred_name='predict', label_name='mos'):
    s_plcc, s_srocc, s_krocc, s_rmse, single_results = calc_content_correlation(results_df, pred_name, label_name)
    plcc_mean, srocc_mean, krocc_mean, rmse_mean = np.mean(s_plcc), np.mean(s_srocc), np.mean(s_krocc), np.mean(s_rmse)
    print(f'Single content average: PLCC: {plcc_mean:.4f} | SROCC: {srocc_mean:.4f} | KROCC: {krocc_mean:.4f} | RMSE: {rmse_mean:.4f}')

    save_path_corr = os.path.join(save_dir, f'{save_name}_{metric}_corr.csv')
    pd.DataFrame(single_results).to_csv(save_path_corr, mode='w', header=True, index=False)




