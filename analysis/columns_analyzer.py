import pandas as pd, numpy as np


class CategoricalColumnAnalyzer:
  def get(column_data : pd.Series):
    return {
        "numberOfCategories" : len(np.unique(column_data)),
        "proportionOfEachCategory" : CategoricalColumnAnalyzer.getEachCategoryProportion(column_data),
        "missingValues" : np.sum(column_data.isnull())
    }
  def getEachCategoryProportion(self, column_data : pd.Series):
      res = {}
      for cat in np.unique(column_data):
        res[cat] = len(column_data[column_data==cat])/column_data.shape[0]
      return res
  
class ContinuousColumnAnalyzer:
    def get(column_data : pd.Series):
        return {
            "min" : column_data.min(),
            "max" : column_data.max(),
            "mean" : column_data.mean(),
            "median" : column_data.median(),
            "quartile_1" : column_data.quantile(q=0.25),
            "quartile_3" :  column_data.quantile(q=0.75),
            "std" : column_data.std(),
            "missingValues": np.sum(column_data.isnull())
        }
    
class ColumnAnalyzer:
  def getDetails(self, data : pd.DataFrame) : 
    res = {}
    for column, col_type in data.dtypes.iteritems():
      if col_type in ['object', 'category', 'string', 'boolean'] : 
        res[column] = CategoricalColumnAnalyzer.get(data[column])
      elif col_type in ['float64']:
         res[column] =  ContinuousColumnAnalyzer.get(data[column])
      else: print(f"Le type de la colonne {column} n'a pas été reconnu dans l'analyse des colonnes et sera ignoré dans le rendu final")
    return res

    