import pandas as pd
import numpy as np
import scipy.stats as st

class makePreprocess():
    def __init__(self):
        self.df = pd.read_csv("./priori/city_priory.csv",sep=';',usecols=[0,1,2,3])
        self.city = np.array(self.df)[:,0]
        self.price_m2 = np.array(self.df)[:,1]
        self.param = np.array(self.df)[:,-2:]
        
    
    def scoreCity(self,df,std=0.1):
        array_pos = np.array(df[['approximate_latitude','approximate_longitude']])
        Coeff = np.array([st.norm.pdf(array_pos[:,0],loc=self.param[i,0],scale=std)*st.norm.pdf(array_pos[:,1],loc=self.param[i,1],scale=std) for i in range(len(self.param))]).T
        Coeff =  Coeff/np.array([np.sum(Coeff,axis=1)]).T
        Score = np.sum(np.array([self.price_m2])*Coeff,axis=1)
        for idx,ci in enumerate(np.array(df['city'])):
            for ydx,cy in enumerate(self.city):
                if ci.__contains__(cy):
                    Score[idx]=self.price_m2[ydx]
        return Score
    

    def scoreSize(self,row):
        if row['property_type'] in ['terrain','terrain à bâtir']: return 0
        elif np.isnan(row['size']) and (~np.isnan(row['land_size'])): return row['land_size']
        elif np.isnan(row['size']): return 0
        else: return row["size"]

        
    def scoreLandSize(self,row):
        if row['property_type'] in ['appartement','chambre','divers','duplex','loft','maison']:
            if np.isnan(row['land_size']): return 0
            else: return row['land_size']
        elif row['property_type'] in ['terrain','terrain à bâtir']:
            if np.isnan(row['land_size']) and (~np.isnan(row['size'])): return row['size']
            elif np.isnan(row['land_size']): return 0
            else: return row['land_size']
        elif np.isnan(row['land_size']): return 0
        else: return row['land_size']
            

    def scoreFloor(self,row):
        if np.isnan(row['floor']) or row['floor']==0: return 0
        else: return np.log((row['floor'])+1)

    
    def scoreRoom(self,row):
        if np.isnan(row['nb_rooms']): return 0
        else: return row['nb_rooms']
                
    def scorePropertyType(self,df):
        def minimizeCategoryPropertyType(value):
            if value in ["château", "atelier", "hôtel particulier", "manoir", "péniche", "villa", "moulin", "loft", "propriété"]:
                return 1.000000
            elif value in ["terrain", "terrain à bâtir"] : return 0.063189
            elif value in ["chalet", "ferme",  "gîte", "viager", "maison"] : return 0.492737
            elif value in ["duplex", "chambre", "appartement"] : return 0.543463
            elif value in ["hôtel","divers"] : return 0.369380
            elif value in ["parking"] : return 0
            else : raise Exception("Property type category not recognized")
    
        return df.property_type.apply(minimizeCategoryPropertyType)
    
    def applyPreprocessing(self,df,test=False):
        df_preprocessed = pd.DataFrame([],index=df.index)
        df_preprocessed['scorePropertyType']=self.scorePropertyType(df)
        df_preprocessed['scoreCity']=self.scoreCity(df)
        df_preprocessed['scoreSize']=df.apply(lambda row : self.scoreSize(row), axis = 1)
        df_preprocessed['scoreLandsize']=df.apply(lambda row : self.scoreLandsize(row), axis = 1)
        df_preprocessed['scoreFloor']=df.apply(lambda row : self.scoreFloor(row), axis=1)
        df_preprocessed['scoreRoom']=df.apply(lambda row : self.scoreRoom(row), axis=1)
        df_preprocessed[['nb_parking_places', 'nb_boxes',
       'has_a_balcony', 'nb_terraces', 'has_a_cellar', 'has_a_garage',
       'has_air_conditioning', 'last_floor', 'upper_floors']]=df[['nb_parking_places', 'nb_boxes',
       'has_a_balcony', 'nb_terraces', 'has_a_cellar', 'has_a_garage',
       'has_air_conditioning', 'last_floor', 'upper_floors']]
        
        if test==False:
            df_preprocessed['price']=df['price']
            
        return df_preprocessed