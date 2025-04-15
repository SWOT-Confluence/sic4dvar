#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIC4DVAR-LC
Copyright (C) 2025 INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from scipy.stats import spearmanr, pearsonr, rankdata
import numpy as np

def pearson_correlation(y_true, y_pred, time_frequency=np.array([])):
    n = len(y_true)
    if time_frequency.any() == 0:
        stat, pvalue = pearsonr(y_true, y_pred)
    else:
        y_true_mean = np.average(y_true, weights=time_frequency)
        y_pred_mean = np.average(y_pred, weights=time_frequency)
        ssd_true = np.sqrt(1 / (n - 1) * np.nansum(time_frequency * (y_true - y_true_mean) ** 2))
        ssd_pred = np.sqrt(1 / (n - 1) * np.nansum(time_frequency * (y_pred - y_pred_mean) ** 2))
        pvalue = None
        stat = 1 / (n - 1) * np.nansum(time_frequency * ((y_true - y_true_mean) / ssd_true) * ((y_pred - y_pred_mean) / ssd_pred))
    return (stat, pvalue)

def spearman_correlation(y_true, y_pred):
    return spearmanr(y_true, y_pred)

def spearman_correlation_coded(y_true, y_pred):
    ranked_y_true = rankdata(y_true)
    ranked_y_pred = rankdata(y_pred)
    diff = rankdata(y_true) - rankdata(y_pred)
    n = len(y_true)
    rs = 1.0 - 6 * np.nansum(diff ** 2) / (n * (n ** 2 - 1))
    return rs

def spearman_correlation_coded2(y_true, y_pred):
    ranked_y_true = rankdata(y_true)
    ranked_y_pred = rankdata(y_pred)
    cov = np.cov(ranked_y_true, ranked_y_pred, bias=True)[0][1]
    rs = cov / (np.std(ranked_y_true) * np.std(ranked_y_pred))
    return rs

def nse(y_true, y_pred, time_frequency=np.array([])):
    if time_frequency.any():
        nse = 1 - np.nansum((y_true - y_pred) ** 2 * time_frequency) / np.nansum((y_true - np.average(y_true, weights=time_frequency)) ** 2 * time_frequency)
    else:
        nse = 1 - np.nansum((y_true - y_pred) ** 2) / np.nansum((y_true - np.nanmean(y_true)) ** 2)
    return nse

def nse1(y_true, y_pred, time_frequency=np.array([])):
    if time_frequency.any():
        nse1 = 1 - np.nansum(abs(y_true - y_pred) * time_frequency) / np.nansum(abs(y_true - np.average(y_true, weights=time_frequency)) * time_frequency)
    else:
        nse1 = 1 - np.nansum(abs(y_true - y_pred)) / np.nansum(abs(y_true - np.nanmean(y_true)))
    return nse1

def nrmse(y_true, y_pred, time_frequency=np.array([])):
    n = len(y_true)
    if time_frequency.any():
        RMSE = np.sqrt(np.average((y_true - y_pred) ** 2, weights=time_frequency))
        y_true_mean = np.average(y_true, weights=time_frequency)
    else:
        RMSE = np.sqrt(np.nansum((y_true - y_pred) ** 2) / n)
        y_true_mean = np.nanmean(y_true)
    return RMSE / y_true_mean

def extrema_high(y_true, y_pred, time_frequency):
    percen_list = [0.99, 0.98, 0.97, 0.96, 0.95]
    y_true_percent_values_list = [np.percentile(y_true, p) for p in percen_list]
    y_pred_percent_values_list = [np.percentile(y_pred, p) for p in percen_list]
    return (np.mean(y_true_percent_values_list) - np.mean(y_pred_percent_values_list)) / np.mean(y_true_percent_values_list)

def extrema_low(y_true, y_pred, time_frequency):
    percen_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    y_true_percent_values_list = [np.percentile(y_true, p) for p in percen_list]
    y_pred_percent_values_list = [np.percentile(y_pred, p) for p in percen_list]
    return (np.mean(y_true_percent_values_list) - np.mean(y_pred_percent_values_list)) / np.mean(y_true_percent_values_list)

def compute_all_indicators_from_predict_true(y_true, y_pred, time_frequency=np.array([])):
    indicators = {}
    indicators['spearman'], *_ = spearman_correlation(y_true, y_pred)
    indicators['nrmse'] = nrmse(y_true, y_pred, time_frequency)
    indicators['nse'] = nse(y_true, y_pred, time_frequency)
    indicators['nnse'] = 1 / (2 - indicators['nse'])
    indicators['pearson'], *_ = pearson_correlation(y_true, y_pred, time_frequency)
    indicators['nbias'] = nbias(y_true, y_pred, time_frequency)
    indicators['absnbias'] = absnbias(y_true, y_pred, time_frequency)
    indicators['nse1'] = nse1(y_true, y_pred, time_frequency)
    indicators['extrema_high'] = extrema_high(y_true, y_pred, time_frequency)
    indicators['extrema_low'] = extrema_low(y_true, y_pred, time_frequency)
    return indicators

def nbias(y_true, y_pred, time_frequency=np.array([])):
    if time_frequency.any():
        y_true_mean = np.nanmean(y_true * time_frequency)
        y_pred_mean = np.nanmean(y_pred * time_frequency)
    else:
        y_true_mean = np.nanmean(y_true)
        y_pred_mean = np.nanmean(y_pred)
    nbias = (y_pred_mean - y_true_mean) / y_true_mean
    return nbias

def absnbias(y_true, y_pred, time_frequency=np.array([])):
    if time_frequency.any():
        y_true_mean = np.nanmean(y_true * time_frequency)
        y_pred_mean = np.nanmean(y_pred * time_frequency)
    else:
        y_true_mean = np.nanmean(y_true)
        y_pred_mean = np.nanmean(y_pred)
    nbias = abs((y_true_mean - y_pred_mean) / y_true_mean)
    return nbias

def main():
    y_true = np.array([2, 3, 4, 5, 6])
    y_pred = np.array([4, 1, 3, 2, 0])
    time_frequency_test = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    indic = compute_all_indicators_from_predict_true(y_true, y_pred)
    indic_time = compute_all_indicators_from_predict_true(y_true, y_pred, time_frequency_test)
    print('indicator value_without_time value_with_time')
    for k in indic.keys():
        print(k, indic[k], indic_time[k])
if __name__ == '__main__':
    main()
