#encoding=utf8
import pandas as pd
import sys

first_file = '../output/single_submission_0.736482_2019-01-21-12-56_folds_1_data.csv'
second_file = '../output/single_submission_0.735968_2019-01-21-19-48_folds_1_data.csv'

def corr(first_file, second_file):
    first_df = pd.read_csv(first_file,index_col=0)
    second_df = pd.read_csv(second_file,index_col=0)
    # assuming first column is `prediction_id` and second column is `prediction`
    prediction = first_df.columns[0]
    # correlation
    print "Finding correlation between: {} and {}".format(first_file,second_file)
    print "Column to be measured: {}".format(prediction)
    print "Pearson's correlation score: {}".format(first_df[prediction].corr(second_df[prediction],method='pearson'))
    print "Kendall's correlation score: {}".format(first_df[prediction].corr(second_df[prediction],method='kendall'))
    print "Spearman's correlation score: {}".format(first_df[prediction].corr(second_df[prediction],method='spearman'))


