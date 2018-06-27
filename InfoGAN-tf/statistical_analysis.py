# -*- coding: utf-8 -*-
"""
Do a statistical analyis on the influence of different hyperparameters on the results.

Created on Tue Jun 26 13:44:29 2018

@author: lbechberger
"""

import sys
import csv
from scipy.stats import shapiro, normaltest, f_oneway, ttest_ind, kruskal, mannwhitneyu

input_file_name = sys.argv[1]
hyperparams = ['ba', 'di', 'la', 'no', 'ep']
ranges = {}
correlations = {}
significance_threshold = 0.05

bins = {'min_range' : ranges, 'overall_cor' : correlations}
for hyperparam in hyperparams:
    for metric, dictionary in bins.items():
        dictionary[hyperparam] = {}

with open(input_file_name, 'r') as in_file:
    reader = csv.DictReader(in_file, delimiter=';')
    
    def try_add(dictionary, key, vector):
        if key not in dictionary.keys():
            dictionary[key] = []
        dictionary[key].append(vector)
       
    for row in reader:
        config_name = row['config']
        
        for hyperparam in hyperparams:
            # find the position of the given hyperparameter
            start_idx = config_name.find(hyperparam)
            # find the end of the given hyperparameter
            end_idx = config_name.find('-', start_idx)
            if config_name[end_idx + 1].isdigit():
                # need special dealing for 'di9e-05'
                end_idx = config_name.find('-', end_idx + 1)
            if end_idx == -1:
                end_idx = len(config_name)
            # figure out the bin name
            bin_name = config_name[(start_idx + len(hyperparam)):end_idx]

            # finally add the data to the corresponding bin
            for metric, dictionary in bins.items():
                try_add(dictionary[hyperparam], bin_name, float(row[metric]))


# now do some statistical tests to check for normal distribution of all the bins
for metric, dictionary in bins.items():
    for hyperparam, bins in dictionary.items():
        
        all_bins_normal = True
        all_bins = []
        
        for value, numbers in bins.items():
            shapiro_stat, shapiro_p = shapiro(numbers)
            d_agostino_stat, d_agostino_p = normaltest(numbers)
            print("Testing normality of '{0}' for {1} = {2} ...".format(metric, hyperparam, value))
            print("\tShapiro test: p = {0} (stat = {1})\t --> {2}".format(shapiro_p, shapiro_stat, (shapiro_p >= significance_threshold)))
            print("\tD'Agostino test: p = {0} (stat = {1})\t --> {2}".format(d_agostino_p, d_agostino_stat, (d_agostino_p >= significance_threshold)))
            
            if len(sys.argv) > 2:
                from matplotlib import pyplot as plt
                plt.hist(numbers, bins=20)
                plt.show()
            
            this_bin_normal = (shapiro_p >= significance_threshold) and (d_agostino_p >= significance_threshold)
            all_bins_normal = all_bins_normal and this_bin_normal
            all_bins.append(numbers)
        
        if all_bins_normal:
            # do ANOVA
            print("\nAll bins normally distributed --> conducting ANOVA")

            anova_stat, anova_p = f_oneway(*all_bins)
            print("ANOVA result: p = {0} (stat = {1})".format(anova_p, anova_stat))
            
            if anova_p < significance_threshold:
                # we found significant differences! use t-test to follow up
                print("ANOVA detected significant differences, using pairwise t-tests to follow up...")
                list_of_values = list(bins.keys())
                # use Bonferroni adjustments
                number_of_comparisons = (len(list_of_values) * (len(list_of_values) - 1)) / 2
                bonferroni_threshold = significance_threshold / number_of_comparisons
                for i in range(len(list_of_values)):
                    for j in range(i + 1, len(list_of_values)):
                        first_name = list_of_values[i]
                        second_name = list_of_values[j]
                        first_numbers = bins[first_name]
                        second_numbers = bins[second_name]
                        t_test_stat, t_test_p = ttest_ind(first_numbers, second_numbers)
                        significance = "SIGNIFICANT" if t_test_p < bonferroni_threshold else "NOT SIGNIFICANT"
                        print("\tDifference between {0} and {1}: p = {2} (stat: {3})\t{4}".format(first_name, second_name, t_test_p, t_test_stat, significance))
        else:
            # do something else
            print("\nAt least one bin not normally distributed --> conducting Kruskal-Wallis")
            kruskal_stat, kruskal_p = kruskal(*all_bins)
            print("Kruskal-Wallis result: p = {0} (stat = {1})".format(kruskal_p, kruskal_stat))
            
            if kruskal_p < significance_threshold:
                # we found significant differences! use Mann-Whitney-U to follow up
                print("Kriskal-Wallis detected significant differences, using pairwise Mann-Whitney-U to follow up...")
                list_of_values = list(bins.keys())
                # use Bonferroni adjustments
                number_of_comparisons = (len(list_of_values) * (len(list_of_values) - 1)) / 2
                bonferroni_threshold = significance_threshold / number_of_comparisons
                for i in range(len(list_of_values)):
                    for j in range(i + 1, len(list_of_values)):
                        first_name = list_of_values[i]
                        second_name = list_of_values[j]
                        first_numbers = bins[first_name]
                        second_numbers = bins[second_name]
                        mannwhitneyu_stat, mannwhitneyu_p = mannwhitneyu(first_numbers, second_numbers)
                        significance = "SIGNIFICANT" if mannwhitneyu_p < bonferroni_threshold else "NOT SIGNIFICANT"
                        print("\tDifference between {0} and {1}: p = {2} (stat: {3})\t{4}".format(first_name, second_name, mannwhitneyu_p, mannwhitneyu_stat, significance))
        print("\n")
    print("\n")
        
# TODO also look for interaction effects?