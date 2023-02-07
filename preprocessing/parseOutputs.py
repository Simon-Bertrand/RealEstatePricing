import os
import pandas as pd


def parseOutputs():
  # extract the results from the csv files without adding the index column
  return pd.concat([pd.read_csv("images_captionning_results/" + f.name) for f in os.scandir("images_captionning_results")], ignore_index=True)
  # return pd.concat([pd.read_csv("images_captionning_results/" + f.name, index_col=0) for f in os.scandir("images_captionning_results")])
  # return pd.concat([ pd.read_csv("images_captionning_results/" + f.name) for f in os.scandir("images_captionning_results")])

