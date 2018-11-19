# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:05:21 2018

@author: lbechberger
"""

import sys
import numpy as np

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]
grouping = ''
if len(sys.argv) > 3:
    grouping = sys.argv[3]

with open(input_file_name, 'r') as in_file:
    with open(output_file_name, 'w') as out_file:
        
        data_points = {}        

        def try_add(config, vector):
            if config not in data_points.keys():
                data_points[config] = []
            data_points[config].append(vector)
           
        def is_float(string):
            try:
                float(string)
                return True
            except ValueError:
                return False

        def try_convert(string):
            if is_float(string):
                return float(string)
            return string
        
        def most_common(column):
            values = set(column)
            most_common_value, counter = None, -1
            for value in values:
                count = column.count(value)
                if (count > counter):
                    counter = count
                    most_common_value = value
            return most_common_value
        
        for line in in_file:
            if (line.startswith("config")):
                # first line - just copy (and add 'counter' column)
                out_line = "{0},{1}\n".format(line.replace("\n", ""), 'counter')
                out_file.write(out_line)
            else:
                # regular line --> add to dictionary
                tokens = line.replace(",\n", '').split(",")
                
                # take care of the grouping if necessary
                if len(grouping) > 0:
                    # find the position of the given hyperparameter
                    start_idx = tokens[0].find(grouping)
                    # find the end of the given hyperparameter
                    end_idx = tokens[0].find('-', start_idx)
                    if tokens[0][end_idx + 1].isdigit():
                        # need special dealing for 'di9e-05'
                        end_idx = tokens[0].find('-', end_idx + 1)
                    if end_idx == -1:
                        end_idx = len(tokens[0])
                    bin_name = tokens[0][start_idx:end_idx]
                else:
                    bin_name = tokens[0]
                
                try_add(bin_name, list(map(lambda x: try_convert(x), tokens[1:])))
        
        for config, lines in data_points.items():
            array = np.array(lines)
            averages = []
            for column in array.T:
                if is_float(column[0]):
                    averages.append(np.mean(np.array(column, dtype="float32")))
                else:
                    averages.append(most_common(list(column)))
            
            out_file.write("{0},{1},{2}\n".format(config, ",".join(map(lambda x: str(x), averages)), len(array)))            