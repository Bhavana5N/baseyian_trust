import pandas as pd
df=pd.read_csv("user1apibaseyianless.csv")
cols = ["robot_beliefA", "robot_actionA", "informant_actionA", "informant_beliefA", \
                     "robot_beliefB", "robot_actionB", "informant_actionB", "informant_beliefB", "Robot Opinion",
                     "Robot Goal Status", "Human Goal Status", "Trust", "Post P"]
data_list= []
for _, row in df.iterrows():
    #print(row)
    if row["Trust"]!=0:
        data_list.append(row["Trust"])

print(data_list)
data_list=data_list[250:]
import plotly.express as px
import plotly.graph_objects as go

import numpy as np

y = list(range(0, len(data_list)))

fig = px.scatter(x=y, y=data_list)
fig.show()

