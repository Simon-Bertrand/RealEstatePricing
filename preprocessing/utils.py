import pandas as pd
import numpy as np
import scipy.stats as st

class preprocessing_score():
    def __init__(self):
        self.df = pd.read_csv("preprocessing/statistic_id744823_square-meter-housing-price-in-france-2022-by-type-of-housing.csv",sep=';',usecols=[0,1,2,3])
        self.city = np.array(self.df)[:,0]
        self.price_m2 = np.array(self.df)[:,1]
        self.param = np.array(self.df)[:,-2:]
        
    
    def ScoreCity(self,df,std=0.1):

        array_pos = np.array(df[['approximate_latitude','approximate_longitude']])
        Coeff = np.array([st.norm.pdf(array_pos[:,0],loc=self.param[i,0],scale=std)*st.norm.pdf(array_pos[:,1],loc=self.param[i,1],scale=std) for i in range(len(self.param))]).T
        Coeff =  Coeff/np.array([np.sum(Coeff,axis=1)]).T
        Score = np.sum(np.array([self.price_m2])*Coeff,axis=1)
        for idx,ci in enumerate(np.array(df['city'])):
            for ydx,cy in enumerate(self.city):
                if ci.__contains__(cy):
                    Score[idx]=self.price_m2[ydx]
        return Score
    
    def ScoreSize(self,row):
        if row['property_type'] in ['appartement','chambre','divers','duplex','loft','maison','villa']:
            if row['property_type'] == 'appartement' and row['size']>300:
                return row['size']/10
            elif row['property_type'] == 'appartement' and row['size']>1000:
                return row['size']/50
            elif row['property_type'] == 'appartement' and row['size']>2000:
                return row['size']/100
            elif row['property_type'] == 'chambre' and row['size']>100:
                return row['size']/10
            elif row['property_type'] == 'divers' and row['size']>10000:
                return row['size']/10
            elif row['property_type'] in ['duplex','loft','maison','villa'] and row['size']>2000:
                return row['size']/10
            elif np.isnan(row['size']) and (~np.isnan(row['land_size'])) and row['land_size']<300:
                return row['land_size']
            elif np.isnan(row['size']) and (np.isnan(row['land_size']) or row['land_size']>=300):
                return 115 #Median Size
            else:
                return row['size']
        elif row['property_type'] in ['terrain','terrain à bâtir']:
            return 0
        elif np.isnan(row['size']) and (~np.isnan(row['land_size'])) and row['land_size']<300: # Attention Ferme ? 
            return row['land_size']
        
        elif np.isnan(row['size']) and (np.isnan(row['land_size']) or row['land_size']>=300):
            return 450 
        else:
            return row["size"]
        
    def ScoreLandsize(self,row):
        if row['property_type'] in ['appartement','chambre','divers','duplex','loft','maison']:
            if np.isnan(row['size']) and ~np.isnan(row['land_size']) and row['land_size']<300:
                return 0
            elif np.isnan(row['land_size']):
                return 0
            else: 
                return row['land_size']
        elif row['property_type'] in ['terrain','terrain à bâtir']:
            if np.isnan(row['land_size']) and (~np.isnan(row['size'])):
                return row['size']
            elif np.isnan(row['land_size']):
                return 4000 #Mean
            else:
                return row['land_size']
        elif np.isnan(row['size']) and ~np.isnan(row['land_size']) and row['land_size']<300:
                return 0
        elif np.isnan(row['land_size']):
                return 0
        else:
            return row['land_size']
            

    def ScoreFloor(self,row):
        if row['property_type'] in ['appartement','chambre','duplex']:
            if row['floor'] > 38: # Highest living building in France
                return 2 #Median of appartement and duplex (chambre = 7)
            elif np.isnan(row['floor']) or row['floor']==0:
                return 0
            else:
                return np.log((row['floor'])+1)
        else:
            return 0
    
    def ScoreRoom(self,row):
        if np.isnan(row['nb_rooms']):
            if (~np.isnan(row['size'])):
                return row['size']/20
            else:
                return 1
        else:
            return row['nb_rooms']
                
    def scorePropertyType(self,df):
        def minimizeCategoryPropertyType(value):
            if value in ["château", "atelier", "hôtel particulier", "manoir", "péniche", "villa", "moulin", "loft", "propriété"]:
                return 'diversCher'
            elif value in ["terrain", "terrain à bâtir"] : return "terrain"
            elif value in ["chalet", "ferme",  "gîte", "viager", "maison"] : return "maison"
            elif value in ["duplex", "chambre", "appartement"] : return "appartement"
            elif value in ["hôtel"] : return "divers"
            elif value in ["parking"] : return "parking"
            elif value in ["divers"] : return "divers"
            else : return value
        def scoringMap(value):
            if value =='appartement': return 0.543463 
            elif value == "divers" : return 0.369380
            elif value == "diversCher" : return 1.000000
            elif value == "maison" : return 0.492737
            elif value == "parking" : return 0.000000
            elif value == "terrain" : return 0.063189
            else:  raise Exception("Property type category not recognized")
        return df.property_type.apply(minimizeCategoryPropertyType).apply(scoringMap)
    
    def apply_preprocessing(self,df,test=False):
        df_preprocessed = pd.DataFrame([],index=df.index)
        df_preprocessed['scorePropertyType']=self.scorePropertyType(df)
        df_preprocessed['scoreCity']=self.ScoreCity(df)
        df_preprocessed['scoreSize']=df.apply(lambda row : self.ScoreSize(row), axis = 1)
        df_preprocessed['scoreLandsize']=df.apply(lambda row : self.ScoreLandsize(row), axis = 1)
        df_preprocessed['scoreFloor']=df.apply(lambda row : self.ScoreFloor(row), axis=1)
        df_preprocessed['scoreRoom']=df.apply(lambda row : self.ScoreRoom(row), axis=1)
        df_preprocessed[['nb_parking_places', 'nb_boxes',
       'has_a_balcony', 'nb_terraces', 'has_a_cellar', 'has_a_garage',
       'has_air_conditioning', 'last_floor', 'upper_floors']]=df[['nb_parking_places', 'nb_boxes',
       'has_a_balcony', 'nb_terraces', 'has_a_cellar', 'has_a_garage',
       'has_air_conditioning', 'last_floor', 'upper_floors']]
        
        if test==False:
            df_preprocessed['price']=df['price']
            
        return df_preprocessed