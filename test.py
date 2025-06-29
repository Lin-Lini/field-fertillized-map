import matplotlib.pyplot as plt
import seaborn as sns
df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['NDVI'], df['EVI'], c=df['cluster'], cmap='viridis')
plt.xlabel('NDVI')
plt.ylabel('EVI')
plt.title('Распределение кластеров по индексу NDVI и EVI')
plt.legend(*scatter.legend_elements(), title="Кластеры")
plt.show()


# Вычисление средних значений почвенных показателей по кластерам
cluster_means = df.groupby('cluster')[['pH', 'GUM', 'K2O']].mean()

# Столбчатая диаграмма средних значений
cluster_means.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Кластер')
plt.ylabel('Средние значения')
plt.title('Средние значения почвенных показателей по кластерам')
plt.xticks(rotation=0)
plt.legend(title='Показатели')
plt.show()


# Диаграмма бокса для pH по кластерам
plt.figure(figsize=(10, 6))
sns.boxplot(x='cluster', y='pH', data=df, palette='viridis')
plt.xlabel('Кластер')
plt.ylabel('pH')
plt.title('Распределение pH по кластерам')
plt.show()
