import os
import pandas as pd


def parseOutputs():
  return pd.concat([ pd.read_csv("images_captionning_results/" + f.name) for f in os.scandir("images_captionning_results")])

