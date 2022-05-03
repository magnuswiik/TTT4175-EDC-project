from matplotlib.axis import YAxis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

plt.rcParams["figure.figsize"] = [15, 3.50]
#plt.rcParams["figure.autolayout"] = True

#headers = ['Alpha','dirt1', 'ErrorRateTraining', 'dirt2','dirt3']
headers = ['Alpha',  'ErrorRateTraining', 'ErrorRateTest']



df = pd.read_csv('iris/worsed_0_features_removed.csv', names=headers)
#df = pd.read_csv('iris/data/worsed_1_feature_removed.csv', names=headers)
#df.drop('dirt1', inplace=True, axis=1)
df.drop('ErrorRateTraining', inplace=True, axis=1)
#df.drop('dirt3', inplace=True, axis=1)

fig = go.Figure(go.Scatter(x = df['Alpha'], y = df['ErrorRateTest'],
                  name='ErrorRateTest'))

fig.update_layout(title='Alpha vs ErrorRateTest',
                   plot_bgcolor='rgb(230, 230,230)', xaxis_title="Alpha",yaxis_title="Error Percent (%)",
                   showlegend=True)

fig.show()

plt.show()