import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="quality", data=df)

count_corr=df.corr()
sns.heatmap(count_corr)

plt.figure(figsize=(10,8))
g=sns.boxplot(data=df.iloc[:,:-1])
g.xaxis.set_tick_params(rotation=90)
