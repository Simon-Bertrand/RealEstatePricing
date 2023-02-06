import pandas as pd

class DataOverview():
    
    def __init__(self,df):
        self.df = df
        
    def NaN_Count(self,name):
        if name in self.df.columns.values:
            print(" Il y a {:.0f} NaN pour {.:0f} soit {:.2f}%" .format(len(self.df[name])-len(self.df[name].dropna()),len(self.df[name]),len(self.df[name].dropna())/len(self.df[name])))
        else:
            raise Exception("{} n'est pas dans la liste. Les choix possibles sont : {}".format(name,self.df.columns.values))