# main.py

from flask import Flask, render_template, request, send_file
import ee
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import shape
import simplekml
import numpy as np
import folium
from folium.plugins import HeatMap
from scipy.interpolate import griddata
import os
import io

app = Flask(__name__)

# Инициализация Earth Engine
def initialize_earth_engine():
    try:
        ee.Initialize(project="rgau-msha-project")
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project="rgau-msha-project")

# Загрузка геометрий (полигонов)
def load_geometry():
    try:
        geometry = ee.FeatureCollection('projects/rgau-msha-project/assets/Polygons')
        geometry_size = geometry.size().getInfo()
        print(f"Количество геометрий в коллекции: {geometry_size}")
        if geometry_size == 0:
            raise ValueError("Коллекция полигонов пуста.")
        return geometry
    except Exception as e:
        print(f"Ошибка при загрузке коллекции полигонов: {e}")
        return None# Загрузка изображений показателей pH, GUM, K2O
def load_soil_indicators():
    try:
        pH = ee.Image('projects/rgau-msha-project/assets/pH')
        gum = ee.Image('projects/rgau-msha-project/assets/GUM')
        k2o = ee.Image('projects/rgau-msha-project/assets/K2O')
        print("Изображения pH, GUM, K2O успешно загружены.")
        return pH, gum, k2o
    except Exception as e:
        print(f"Ошибка при загрузке показателей pH, GUM, K2O: {e}")
        return None, None, None

# Функция для расчёта индексов NDVI, EVI, NDWI
def add_indices(image):
    try:
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }).rename('EVI')
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return image.addBands([ndvi, evi, ndwi])
    except Exception as e:
        print(f"Ошибка при добавлении индексов: {e}")
        return image

# Создание временных композитов (медианных за месяц)
def create_time_composites(collection, start_year, end_year, geometry, expected_indices):
    time_composites = []
    for year in range(start_year, end_year + 1):
        for month in range(5, 10):  # Месяцы с 5 по 9 включительно (май-сентябрь)
            start_date = f'{year}-{month:02d}-01'
            if month in [1, 3, 5, 7, 8, 10, 12]:
                end_day = 31
            elif month == 2:
                end_day = 28
            else:
                end_day = 30
            end_date = f'{year}-{month:02d}-{end_day:02d}'
            monthly_collection = collection.filterDate(start_date, end_date) \
                                          .filterBounds(geometry) \
                                          .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 30)
            monthly_size = monthly_collection.size().getInfo()
            print(f"Месяц {year}-{month:02d}: {monthly_size} изображений")
            if monthly_size == 0:
                print(f"Предупреждение: В месяце {year}-{month:02d} нет доступных изображений после фильтрации.")
                continue
            monthly_composite = monthly_collection.median().set({'year': year, 'month': month})
            bands_monthly = monthly_composite.bandNames().getInfo()
            missing_indices_monthly = [idx for idx in expected_indices if idx not in bands_monthly]
            if missing_indices_monthly:
                print(f"Отсутствуют индексы в месячном композитном изображении {year}-{month:02d}: {missing_indices_monthly}")
            else:
                print(f"Все индексы присутствуют в месячном композитном изображении {year}-{month:02d}.")
            time_composites.append(monthly_composite)
    return ee.ImageCollection(time_composites)

# Функция для выполнения всего анализа
def run_analysis():
    # Инициализация
    initialize_earth_engine()
    
    # Загрузка данных
    geometry = load_geometry()
    if geometry is None:
        return {"error": "Не удалось загрузить геометрию полигонов."}
    
    pH, gum, k2o = load_soil_indicators()
    
    # Загрузка и фильтрация коллекции изображений Sentinel-2
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate('2020-05-01', '2024-09-30') \
            .filterBounds(geometry) \
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 30)
        total_size = collection.size().getInfo()
        print(f"Общее количество изображений в коллекции: {total_size}")
    except Exception as e:
        print(f"Ошибка при загрузке коллекции изображений Sentinel-2: {e}")
        return {"error": "Не удалось загрузить коллекцию изображений Sentinel-2."}
    
    if total_size == 0:
        return {"error": "Коллекция изображений пуста."}
    
    # Добавление индексов
    print("Расчёт индексов NDVI, EVI, NDWI...")
    indexed_collection = collection.map(add_indices)
    first_indexed_image = indexed_collection.first()
    first_indexed_bands = first_indexed_image.bandNames().getInfo()
    expected_indices = ['NDVI', 'EVI', 'NDWI']
    missing_indices = [idx for idx in expected_indices if idx not in first_indexed_bands]
    if missing_indices:
        print(f"Отсутствуют индексы в первом индексированном изображении: {missing_indices}")
        return {"error": f"Отсутствуют индексы: {missing_indices}"}
    else:
        print("Все индексы присутствуют в первом индексированном изображении.")
    
    # Создание временных композитов
    print("Создание временных композитов...")
    time_composites = create_time_composites(indexed_collection, 2020, 2024, geometry, expected_indices)
    composites_size = time_composites.size().getInfo()
    print(f"Количество временных композитов: {composites_size}")
    
    if composites_size == 0:
        return {"error": "Нет временных композитов для анализа."}
    
    # Создание медианного изображения
    print("Создание среднего изображения по всем композитам с использованием медианы...")
    try:
        median_image = time_composites.median()
        median_image_bands = median_image.bandNames().getInfo()
        print(f"Банды медианного изображения: {median_image_bands}")
    except Exception as e:
        print(f"Ошибка при создании медианного изображения: {e}")
        return {"error": "Не удалось создать медианное изображение."}
    
    # Проверка наличия индексов в медианном изображении
    print("Проверка наличия индексов в медианном композитном изображении...")
    try:
        bands_median_image = median_image.bandNames().getInfo()
        print(f"Банды в медианном композитном изображении: {bands_median_image}")
        missing_indices_median = [idx for idx in expected_indices if idx not in bands_median_image]
        if missing_indices_median:
            print(f"Отсутствуют индексы в медианном композитном изображении: {missing_indices_median}")
            return {"error": f"Отсутствуют индексы в медианном изображении: {missing_indices_median}"}
        else:
            print("Все индексы присутствуют в медианном композитном изображении.")
    except Exception as e:
        print(f"Ошибка при проверке индексов в медианном композитном изображении: {e}")
        return {"error": "Не удалось проверить индексы в медианном композитном изображении."}
    
    # Добавление почвенных показателей pH, GUM, K2O
    if pH and gum and k2o:
        print("Добавление почвенных показателей pH, GUM, K2O к медианному изображению...")
        try:
            first_image = indexed_collection.first()
            median_proj = first_image.select('B1').projection()
            median_scale = first_image.select('B1').projection().nominalScale().getInfo()
            print(f"Проекция медианного композиту: {median_proj.getInfo()}")
            print(f"Разрешение медианного композиту: {median_scale} метров")
            
            pH_reproj = pH.reproject(crs=median_proj, scale=median_scale).rename('pH')
            gum_reproj = gum.reproject(crs=median_proj, scale=median_scale).rename('GUM')
            k2o_reproj = k2o.reproject(crs=median_proj, scale=median_scale).rename('K2O')
            
            # Проверка наличия данных в перепроецированных слоях
            pH_stats = pH_reproj.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=geometry,
                scale=median_scale,
                maxPixels=1e13
            ).getInfo()
            print(f"Статистика pH после перепроецирования: {pH_stats}")
            
            gum_stats = gum_reproj.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=geometry,
                scale=median_scale,
                maxPixels=1e13
            ).getInfo()
            print(f"Статистика GUM после перепроецирования: {gum_stats}")
            
            k2o_stats = k2o_reproj.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=geometry,
                scale=median_scale,
                maxPixels=1e13
            ).getInfo()
            print(f"Статистика K2O после перепроецирования: {k2o_stats}")
            
            # Добавление почвенных показателей к медианному изображению
            stacked_image = median_image.addBands([pH_reproj, gum_reproj, k2o_reproj])
            
            # Проверка наличия индексов и почвенных показателей
            print("Проверка наличия индексов и почвенных показателей в stacked_image...")
            bands_stacked_image = stacked_image.bandNames().getInfo()
            print(f"Банды в stacked_image: {bands_stacked_image}")
            missing_indices_stacked = [idx for idx in expected_indices + ['pH', 'GUM', 'K2O'] if idx not in bands_stacked_image]
            if missing_indices_stacked:
                print(f"Отсутствуют индексы или почвенные показатели в stacked_image: {missing_indices_stacked}")
                return {"error": f"Отсутствуют индексы или почвенные показатели: {missing_indices_stacked}"}
            else:
                print("Все индексы и почвенные показатели присутствуют в stacked_image.")
        except Exception as e:
            print(f"Ошибка при добавлении почвенных показателей: {e}")
            print("Продолжаем без почвенных показателей.")
            stacked_image = median_image
    else:
        print("Некоторые почвенные показатели отсутствуют. Продолжаем без них.")
        stacked_image = median_image
    
    # Добавление уникальных идентификаторов к полигонам
    print("Добавление уникальных идентификаторов к полигонам...")
    try:
        list_of_features = geometry.toList(geometry.size())
        features_list = list_of_features.getInfo()
        for idx, feature in enumerate(features_list, start=1):
            if 'properties' in feature:
                feature['properties']['poly_id'] = idx
            else:
                feature['properties'] = {'poly_id': idx}
        indexed_geometry = ee.FeatureCollection(features_list)
    except Exception as e:
        print(f"Ошибка при присвоении poly_id: {e}")
        return {"error": "Не удалось присвоить poly_id."}
    
    # Извлечение средних значений индексов и почвенных показателей для каждого полигона
    print("Извлечение средних значений индексов и почвенных показателей для каждого полигона...")
    try:
        mean_values = stacked_image.reduceRegions(
            collection=indexed_geometry,
            reducer=ee.Reducer.mean(),
            scale=median_scale
        )
    except Exception as e:
        print(f"Ошибка при извлечении средних значений с использованием reduceRegions: {e}")
        return {"error": "Не удалось извлечь средние значения."}
    
    # Получение результатов
    print("Получение результатов из Earth Engine...")
    try:
        mean_values_info = mean_values.getInfo()['features']
        print(f"Количество полученных полигонов с данными: {len(mean_values_info)}")
    except Exception as e:
        print(f"Ошибка при получении данных: {e}")
        mean_values_info = []
    
    if not mean_values_info:
        return {"error": "Нет данных для дальнейшей обработки."}
    
    # Преобразование данных в pandas DataFrame
    print("Преобразование данных в pandas DataFrame...")
    data = []
    for feature in mean_values_info:
        props = feature['properties']
        data.append({
            'poly_id': props.get('poly_id'),
            'NDVI': props.get('NDVI'),
            'EVI': props.get('EVI'),
            'NDWI': props.get('NDWI'),  
            'pH': props.get('pH'),
            'GUM': props.get('GUM'),
            'K2O': props.get('K2O')
        })
    
    df = pd.DataFrame(data)
    print("Пример данных:")
    print(df.head())
    
    print("\nСтатистика по DataFrame:")
    print(df.describe())
    
    print("\nКоличество пропущенных значений в каждой колонке:")
    print(df.isnull().sum())
    
    # Удаление полигонов с отсутствующими значениями NDVI и EVI
    print("\nУдаление полигонов с отсутствующими значениями NDVI и EVI...")
    initial_count = len(df)
    df = df.dropna(subset=['NDVI', 'EVI'], how='all')
    final_count = len(df)
    print(f"Количество полигонов до удаления: {initial_count}")
    print(f"Количество полигонов после удаления: {final_count}")
    
    if final_count == 0:
        return {"error": "Все полигоны были удалены после фильтрации."}
    
    # Кластеризация полигонов
    print("Кластеризация полигонов на основе NDVI, EVI, NDWI, pH, GUM, K2O...")
    try:
        clustering_features = df[['NDVI', 'EVI', 'NDWI', 'pH', 'GUM', 'K2O']].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clustering_features)
    except Exception as e:
        print(f"Ошибка при подготовке данных для кластеризации: {e}")
        return {"error": "Не удалось подготовить данные для кластеризации."}
    
    # Определение числа кластеров с помощью метода "локтя"
    print("Определение оптимального числа кластеров с помощью метода 'локтя'...")
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        try:
            kmeans = KMeans(n_clusters=8, n_init=10, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        except Exception as e:
            print(f"Ошибка при кластеризации с k={k}: {e}")
            inertia.append(None)
    
    # Сохранение графика метода "локтя" в буфер
    plt.figure(figsize=(8, 4))
    plt.plot(K_range, inertia, 'bx-')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Inertia')
    plt.title('Метод локтя для определения числа кластеров')
    plt.xticks(K_range)
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    elbow_chart = buf.getvalue()
    buf.close()
    
    plt.close()
    
    # Выбор оптимального числа кластеров 
    optimal_k = 8
    print(f"Выбор оптимального числа кластеров: {optimal_k}")
    try:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        # Добавление меток кластеров обратно в DataFrame
        df.loc[clustering_features.index, 'cluster'] = labels
        # Преобразование меток кластеров в целые числа
        df['cluster'] = df['cluster'].astype(int)
    except Exception as e:
        print(f"Ошибка при применении KMeans: {e}")
        return {"error": "Не удалось применить KMeans."}
    
    # Пример кластеризации
    print("Пример кластеризации:")
    print(df[['poly_id', 'cluster']].head())
    
    # Определение оптимальных значений для каждого кластера
    print("Определение оптимальных значений для каждого кластера на основе полигона с самым высоким NDVI...")
    optimal_values = {}
    for cluster in range(optimal_k):
        cluster_df = df[df['cluster'] == cluster]
        if not cluster_df.empty:
            max_ndvi_row = cluster_df.loc[cluster_df['NDVI'].idxmax()]
            optimal_values[cluster] = {
                'pH': max_ndvi_row['pH'],
                'GUM': max_ndvi_row['GUM'],
                'K2O': max_ndvi_row['K2O']
            }
            print(f"Кластер {cluster}: Оптимальные значения (poly_id={max_ndvi_row['poly_id']}): {optimal_values[cluster]}")
        else:
            print(f"Кластер {cluster} пуст.")
            optimal_values[cluster] = {'pH': np.nan, 'GUM': np.nan, 'K2O': np.nan}
    
    # Вычисление отклонений от оптимальных значений
    print("Вычисление отклонений от оптимальных значений...")
    def calculate_deviation(row):
        cluster = row['cluster']
        optimal = optimal_values.get(cluster, {'pH': np.nan, 'GUM': np.nan, 'K2O': np.nan})
        row['pH_Deviation'] = optimal['pH'] - row['pH'] if not pd.isna(optimal['pH']) else np.nan
        row['GUM_Deviation'] = optimal['GUM'] - row['GUM'] if not pd.isna(optimal['GUM']) else np.nan
        row['K2O_Deviation'] = optimal['K2O'] - row['K2O'] if not pd.isna(optimal['K2O']) else np.nan
        return row
    
    df = df.apply(calculate_deviation, axis=1)
    
    # Генерация рекомендаций на основе отклонений
    print("Генерация рекомендаций по внесению удобрений...")
    def generate_recommendations(row):
        recommendations = []
        
        # Рекомендации по pH
        if pd.notnull(row['pH_Deviation']):
            if row['pH_Deviation'] > 1.0:
                recommendations.append(f"pH: увеличить на {row['pH_Deviation']:.2f} единиц путем внесения известкования.")
            elif 0.5 < row['pH_Deviation'] <= 1.0:
                recommendations.append(f"pH: умеренно увеличить на {row['pH_Deviation']:.2f} единиц.")
            elif 0 < row['pH_Deviation'] <= 0.5:
                recommendations.append(f"pH: немного увеличить на {row['pH_Deviation']:.2f} единиц.")
            elif row['pH_Deviation'] < -1.0:
                recommendations.append(f"pH: снизить на {abs(row['pH_Deviation']):.2f} единиц путем внесения сероводорода.")
            elif -1.0 <= row['pH_Deviation'] < -0.5:
                recommendations.append(f"pH: умеренно снизить на {abs(row['pH_Deviation']):.2f} единиц.")
            elif -0.5 <= row['pH_Deviation'] < 0:
                recommendations.append(f"pH: немного снизить на {abs(row['pH_Deviation']):.2f} единиц.")
        
        # Рекомендации по GUM
        if pd.notnull(row['GUM_Deviation']):
            if row['GUM_Deviation'] > 1.0:
                recommendations.append(f"GUM: увеличить на {row['GUM_Deviation']:.2f} единиц путем внесения соответствующих удобрений.")
            elif 0.5 < row['GUM_Deviation'] <= 1.0:
                recommendations.append(f"GUM: умеренно увеличить на {row['GUM_Deviation']:.2f} единиц.")
            elif 0 < row['GUM_Deviation'] <= 0.5:
                recommendations.append(f"GUM: немного увеличить на {row['GUM_Deviation']:.2f} единиц.")
            elif row['GUM_Deviation'] < -1.0:
                recommendations.append(f"GUM: снизить на {abs(row['GUM_Deviation']):.2f} единиц.")
            elif -1.0 <= row['GUM_Deviation'] < -0.5:
                recommendations.append(f"GUM: умеренно снизить на {abs(row['GUM_Deviation']):.2f} единиц.")
            elif -0.5 <= row['GUM_Deviation'] < 0:
                recommendations.append(f"GUM: немного снизить на {abs(row['GUM_Deviation']):.2f} единиц.")
        
        # Рекомендации по K2O
        if pd.notnull(row['K2O_Deviation']):
            if row['K2O_Deviation'] > 20:
                recommendations.append(f"K2O: увеличить на {row['K2O_Deviation']:.2f} кг/га путем внесения калийных удобрений.")
            elif 10 < row['K2O_Deviation'] <= 20:
                recommendations.append(f"K2O: умеренно увеличить на {row['K2O_Deviation']:.2f} кг/га.")
            elif 0 < row['K2O_Deviation'] <= 10:
                recommendations.append(f"K2O: немного увеличить на {row['K2O_Deviation']:.2f} кг/га.")
            elif row['K2O_Deviation'] < -20:
                recommendations.append(f"K2O: снизить на {abs(row['K2O_Deviation']):.2f} кг/га.")
            elif -20 <= row['K2O_Deviation'] < -10:
                recommendations.append(f"K2O: умеренно снизить на {abs(row['K2O_Deviation']):.2f} кг/га.")
            elif -10 <= row['K2O_Deviation'] < 0:
                recommendations.append(f"K2O: немного снизить на {abs(row['K2O_Deviation']):.2f} кг/га.")
        
        if not recommendations:
            return "Нет рекомендаций."
        return "; ".join(recommendations)
    
    # Добавление рекомендаций в DataFrame
    df['recommendations'] = df.apply(generate_recommendations, axis=1)
    
    # Проверка результатов
    print("Пример данных с рекомендациями:")
    print(df[['poly_id', 'pH', 'GUM', 'K2O', 
              'pH_Deviation', 'GUM_Deviation', 'K2O_Deviation', 'recommendations']].head())
    
    # Создание GeoDataFrame для экспорта в KML
    print("Создание GeoDataFrame для экспорта в KML...")
    try:
        polygons = indexed_geometry.getInfo()['features']
        
        poly_ids = []
        poly_geometries = []
        for poly in polygons:
            poly_id = poly['properties']['poly_id']
            geom = poly['geometry']
            poly_ids.append(poly_id)
            poly_geometries.append(shape(geom))
        
        gdf = gpd.GeoDataFrame({
    	'poly_id': poly_ids,
    	'geometry': poly_geometries
	}, crs='EPSG:4326')  # Укажите корректный CRS, например, WGS84

        
        # Объединение с данными рекомендаций
        final_df = pd.merge(df, gdf, on='poly_id', how='left')
        
        # Преобразование меток кластеров в целые числа
        final_df['cluster'] = final_df['cluster'].astype(int)
        
        # Создание GeoDataFrame с координатами центроидов
        gdf_geo_proj = gpd.GeoDataFrame(final_df, geometry='geometry', crs='EPSG:32652')
        gdf_geo_proj['centroid'] = gdf_geo_proj.geometry.centroid
        
        # Преобразование центроидов обратно в географическую систему координат
        gdf_geo = gdf_geo_proj.copy()
        gdf_geo['centroid_4326'] = gdf_geo_proj['centroid'].to_crs(epsg=4326)
        gdf_geo['longitude'] = gdf_geo['centroid_4326'].x
        gdf_geo['latitude'] = gdf_geo['centroid_4326'].y
        
        # Обновление final_df с координатами
        final_df['longitude'] = gdf_geo['longitude']
        final_df['latitude'] = gdf_geo['latitude']
        
        print("Пример GeoDataFrame с рекомендациями и координатами:")
        print(final_df[['poly_id', 'longitude', 'latitude', 'pH', 'GUM', 'K2O', 
                       'pH_Deviation', 'GUM_Deviation', 'K2O_Deviation', 'recommendations']].head())
    except Exception as e:
        print(f"Ошибка при создании GeoDataFrame: {e}")
        return {"error": "Не удалось создать GeoDataFrame."}
    
    # Определение категорий и цветов на основе отклонений
    def classify_deviation(row):
        deviations = {
            'pH': abs(row['pH_Deviation']) if pd.notnull(row['pH_Deviation']) else 0,
            'GUM': abs(row['GUM_Deviation']) if pd.notnull(row['GUM_Deviation']) else 0,
            'K2O': abs(row['K2O_Deviation']) if pd.notnull(row['K2O_Deviation']) else 0
        }
        max_dev = max(deviations.values())
        
        if max_dev > 20:
            category = "Критически высокое отклонение"
            color = 'darkred'
        elif 15 < max_dev <= 20:
            category = "Существенное отклонение"
            color = 'red'
        elif 10 < max_dev <= 15:
            category = "Высокое отклонение"
            color = 'orange'
        elif 5 < max_dev <= 10:
            category = "Среднее отклонение"
            color = 'yellow'
        elif 2 < max_dev <= 5:
            category = "Низкое отклонение"
            color = 'lightgreen'
        elif 0 < max_dev <= 2:
            category = "Несущественное отклонение"
            color = 'green'
        else:
            category = "Норма"
            color = 'darkgreen'
        
        return pd.Series([category, color])
    
    print("Классификация отклонений и назначение цветов...")
    final_df[['category', 'color']] = final_df.apply(classify_deviation, axis=1)
    print("Пример данных с категориями и цветами:")
    print(final_df[['poly_id', 'category', 'color']].head())

    # Создание KML-файла
    print("Создание KML-файла с рекомендациями и цветами...")
    try:
        kml = simplekml.Kml()

        for idx, row in final_df.iterrows():
            geom = row['geometry']
            if geom.geom_type == 'Polygon':
                coords = list(geom.exterior.coords)
            elif geom.geom_type == 'MultiPolygon':
                coords = []
                for poly in geom:
                    coords.extend(list(poly.exterior.coords))
            else:
                continue  # Пропустить неподдерживаемые геометрии

            pol = kml.newpolygon(
                name=f"Poly_ID: {row['poly_id']}",
                description=row['recommendations'],
                outerboundaryis=[(x, y) for x, y in coords]
            )

            color_map = {
                'Критически высокое отклонение': 'ff8b0000',  # DarkRed
                'Существенное отклонение': 'ffff0000',  # Red
                'Высокое отклонение': 'ffffa500',  # Orange
                'Среднее отклонение': 'ffffff00',  # Yellow
                'Низкое отклонение': 'ff90ee90',  # LightGreen
                'Несущественное отклонение': 'ff008000',  # Green
                'Норма': 'ff006400'  # DarkGreen
            }
            pol.style.polystyle.color = color_map.get(row['category'], 'ff000000')  # Black по умолчанию
            pol.style.polystyle.outline = 1  # Обводка полигона

            # Добавление отклонений как дополнительные атрибуты в KML
            pol.extendeddata.newdata(name="pH_Deviation", value=f"{row['pH_Deviation']:.2f}")
            pol.extendeddata.newdata(name="GUM_Deviation", value=f"{row['GUM_Deviation']:.2f}")
            pol.extendeddata.newdata(name="K2O_Deviation", value=f"{row['K2O_Deviation']:.2f}")

        # Временный путь для KML-файла
        temp_kml_path = os.path.join("static", "kml", "fertilizer_temp.kml")
        os.makedirs(os.path.dirname(temp_kml_path), exist_ok=True)

        # Сохранение KML на диск
        kml.save(temp_kml_path)

        # Загрузка KML в буфер
        with open(temp_kml_path, "rb") as f:
            kml_buffer = io.BytesIO(f.read())
    except Exception as e:
        print(f"Ошибка при создании KML-файла: {e}")
        kml_buffer = None

    # Создание интерактивной карты с Folium
    print("Создание интерактивной карты с рекомендациями и легендой...")
    try:
        # Создание карты с применением стилей
        central_point = [61.526761, 129.169331]  
        m = folium.Map(
            location=central_point, 
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Определение цветов для кластеров
        cluster_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
        
        # Добавление полигонов на карту
        for idx, row in final_df.iterrows():
            geom = row['geometry']
            if geom.geom_type == 'Polygon':
                coordinates = [(y, x) for x, y in geom.exterior.coords] 
            elif geom.geom_type == 'MultiPolygon':
                # Обработка мультиполигонов, если есть
                coordinates = []
                for poly in geom:
                    coordinates.append([(y, x) for x, y in poly.exterior.coords])
            else:
                continue  # Пропустить неподдерживаемые геометрии
            
            # Определение цвета
            color = row['color']
            
            # Добавление полигона на карту
            folium.Polygon(
                locations=coordinates if geom.geom_type == 'MultiPolygon' else coordinates,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"Poly ID: {row['poly_id']}<br>Recommendations: {row['recommendations']}<br>Category: {row['category']}", 
                    max_width=300
                )
            ).add_to(m)
        
        # Создание карты и легенды
        legend_html = '''
        <div style="
        position: fixed; 
        top: 20px; left: 50px; width: 280px; height: 250px; 
        border: 2px solid grey; z-index:9999; font-size:14px;
        background-color:white;
        padding: 10px;
        overflow: auto;
        border-radius: 8px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);">
        <b>Категории</b><br>
        <i style="background:darkred;width:10px;height:10px;display:inline-block;"></i>&nbsp;Критически низкая урожайность<br>
        <i style="background:red;width:10px;height:10px;display:inline-block;"></i>&nbsp;Очень низкая урожайность<br>
        <i style="background:orange;width:10px;height:10px;display:inline-block;"></i>&nbsp;Низкая урожайность<br>
        <i style="background:yellow;width:10px;height:10px;display:inline-block;"></i>&nbsp;Средняя урожайность<br>
        <i style="background:lightgreen;width:10px;height:10px;display:inline-block;"></i>&nbsp;Высокая урожайность<br>
        <i style="background:green;width:10px;height:10px;display:inline-block;"></i>&nbsp;Очень высокая урожайность<br>
        <i style="background:darkgreen;width:10px;height:10px;display:inline-block;"></i>&nbsp;Идеальная урожайность<br>
        </div>
        ''' 
        m.get_root().html.add_child(folium.Element(legend_html))

        
        # Сохранение карты в буфер
        map_html = m._repr_html_()
    except Exception as e:
        print(f"Ошибка при создании интерактивной карты: {e}")
        map_html = None
    
    # Возвращение результатов
    return {
        "df": final_df.to_html(classes='table table-striped', index=False),
        "map_html": map_html,
        "kml_buffer": kml_buffer
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_analysis', methods=['POST'])
def analysis():
    results = run_analysis()
    if "error" in results:
        return render_template('index.html', error=results["error"])
    
    kml_filename = "fertilizer_recommendations.kml"
    kml_path = os.path.join('static', 'kml')
    os.makedirs(kml_path, exist_ok=True)
    
    if results["kml_buffer"] is not None:
        with open(os.path.join(kml_path, kml_filename), 'wb') as f:
            f.write(results["kml_buffer"].getbuffer())
    else:
        return render_template('index.html', error="Ошибка при создании KML-файла.")

    return render_template('map.html', table=results["df"], map_html=results["map_html"], kml_file=f'kml/{kml_filename}')

@app.route('/download_kml')
def download_kml():
    kml_path = os.path.join('static', 'kml', 'fertilizer_recommendations.kml')
    if os.path.exists(kml_path):
        return send_file(kml_path, as_attachment=True)
    else:
        return "KML-файл не найден.", 404

if __name__ == '__main__':
    app.run(debug=True)