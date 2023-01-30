import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def showDistribPlot(pdData,column,c='property_type'):
    fig = px.scatter(pdData, x=np.arange(0,len(pdData.index),1), y=column,title = column,color = c)
    fig.show()

def histogram(pdData,column,nb=100):
    if pdData[column].dtype == 'O':
        fig = px.histogram(pdData, x=column,nbins = nb,title='Histogram of '+column,color=column)
    else:
        fig = px.histogram(pdData, x=column,nbins = nb,title='Histogram of '+column)
    fig.update_layout(bargap=0.2)
    fig.show()

def corr_matrix(pdData):
    df_corr = pdData.corr()
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x = df_corr.columns,y = df_corr.index,z = np.array(df_corr),text=df_corr.values,texttemplate='%{text:.2f}'))
    fig.show()
