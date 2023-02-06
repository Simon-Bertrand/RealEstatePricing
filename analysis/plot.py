import plotly.express as px
import plotly.graph_objects as go
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class Plot:
    def showDistribPlot(pdData,column,c='property_type',**args):
        fig = px.scatter(pdData, x=np.arange(0,len(pdData.index),1), y=column,title = column,color = c,**args)
        fig.show()

    def histogram(pdData,column,nb=100,**args):
        if pdData[column].dtype == 'O':
            fig = px.histogram(pdData, x=column,nbins = nb,title='Histogram of '+column,color=column,**args)
        else:
            fig = px.histogram(pdData, x=column,nbins = nb,title='Histogram of '+column,**args)
        fig.update_layout(bargap=0.2)
        fig.show()

    def corr_matrix(pdData):
        df_corr = pdData.corr()
        fig = go.Figure()
        fig.add_trace(go.Heatmap(x = df_corr.columns,y = df_corr.index,z = np.array(df_corr),text=df_corr.values,texttemplate='%{text:.2f}'))
        fig.show()

    def proportionMissingValuesHeatmap(df, subpopulate):
        df_groupped = df.groupby([subpopulate]).agg(lambda x:np.sum(x.isna())/len(x))
        mask = (df_groupped.sum(axis=0) != 0)
        mask_cols = mask[mask].index.tolist()
        df_groupped[mask_cols]
        fig, ax = plt.subplots(figsize=(10,10))  
        ax.set_title(f"Proportion of missing value in the sub-population of '{subpopulate}'")
        s=sns.heatmap(df_groupped[mask_cols], annot=True)
        s.set_xlabel('Features (columns) of dataframe')
        s.set_ylabel(subpopulate)
        plt.savefig(f"analysis/images/missingvalues_by_{subpopulate}.jpg")

    def subPairPlot(df, subpopulate, hue):
        df_sampled = df.sample(n=1000)
        sns.pairplot(df_sampled , x_vars="price", hue=hue, height=2, aspect=6, diag_kind="hist")
        plt.savefig(f"analysis/images/subPairPlot_by_{subpopulate}_on_{hue}.jpg")