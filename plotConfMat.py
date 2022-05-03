import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


flowers = ["Predicted Setosa", "Predicted Versicolor","Predicted Virginic"]
flowers2 = ["Actual Setosa", "Actual Versicolor","Actual Virginic"]


array = [[30.,  0.,  0.],
 [ 0., 27.,  3.],
 [ 0.,  1., 29.]]
df_cm = pd.DataFrame(array, index = [i for i in flowers2],
                  columns = [i for i in flowers])
plt.figure(figsize = (10,7))
plt.title('Confusion Matrix for train-set with Sepal Width, Sepal Length and Petal Length removed')
sn.heatmap(df_cm, annot=True)
plt.show()