import pandas as pd, numpy as np


class CategoricalColumnAnalyzer:
  def get(column_data : pd.Series):
    return {
        "numberOfCategories" : len(np.unique(column_data.dropna())),
        "proportionOfEachCategory" : CategoricalColumnAnalyzer.getEachCategoryProportion(column_data),
        "missingValues" : np.sum(column_data.isnull())
    }
  def getEachCategoryProportion(column_data : pd.Series):
      res = {}
      for cat in np.unique(column_data.dropna()):
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
  def getDetails(self, data : pd.DataFrame, ignoring = []) : 
    try : return self.data
    except:
      res = {}
      for column, col_type in data.dtypes.iteritems():
        if column in ignoring: continue
        if col_type in ['object', 'category', 'string', 'boolean'] : 
          res[column] = CategoricalColumnAnalyzer.get(data[column])
          res[column]['col_type'] = str(col_type)

        elif col_type in ['float64']:
          res[column] =  ContinuousColumnAnalyzer.get(data[column])
          res[column]['col_type'] = str(col_type)

        else: print(f"Le type de la colonne {column} n'a pas été reconnu dans l'analyse des colonnes et sera ignoré dans le rendu final")
      self.data = res
      return self.data
    
  def prettyPrint(self):
      s=""
      for col in self.data.keys():
        s += f"{25*'_'}\n{col}:\n"
        sub_string = ""
        if isinstance(self.data[col], dict):
          for key, val in self.data[col].items():
            if key == "proportionOfEachCategory" and len(val.keys()) > 25:
              continue
            sub_string += f"[{key}] = {np.round(val,4) if isinstance(val, float) else val} ; \n"
          s+=sub_string + "\n"
        else :
          s += f"[{col}] = {self.data[col]} ;"
      print(s)
