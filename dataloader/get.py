

from dataloader.utils import DataChecker
import pandas as pd, os, numpy as np
from skimage import io
class DataGetter():
    def __init__(self):
        self.data_folder_loc = DataChecker.getDataFolder()

    def readTabular(self):
        try: return self.tabular
        except : 
            self.tabular =  {
                "X_test" : pd.read_csv(self.data_folder_loc+"/X_test_BEhvxAN.csv", index_col="id_annonce"),
                "X_train" : pd.read_csv(self.data_folder_loc+"/X_train_J01Z4CN.csv",index_col="id_annonce"),
                "y_train" : pd.read_csv(self.data_folder_loc+"/y_train_OXxrJt1.csv",index_col="id_annonce"),
            }
            return self.tabular
    
    def getData(self, test_or_train):
        if not test_or_train in ['test', 'train'] : 
            raise Exception("Le deuxième argument de getData doit être 'train' ou 'test'")
        df = self.readTabular()['X_'+test_or_train]
        if test_or_train == "train":
            df['price'] = self.readTabular()['y_train']['price']
        df['images'] = self.readTabular()['X_'+test_or_train].index.map(lambda x: LazyImages(self.data_folder_loc+"/reduced_images", test_or_train, x))
        return df

    def iterateLoadedData(self, test_or_train):
        return (
            (idx, data[(data.index.values != "images")&(data.index.values != "price")], data.images.load(), data.price)
            for idx, data in self.getData(test_or_train).iterrows()
        )


class LazyImages():
    def __init__(self, folder_loc, test_or_train, id_annonce):
        self.paths = [f.path for f in os.scandir(folder_loc+"/"+test_or_train+"/ann_"+str(id_annonce))]
        self.data = np.NaN


    def load(self):
        self.data = [io.imread(p) for p in self.paths]
        return self.data

    def isLoaded(self):
        return self.data == np.NaN

    def __repr__(self):
        return f"LazyImages(len={len(self.paths)}, isLoaded={self.isLoaded()})"


