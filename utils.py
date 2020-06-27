import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def generate_barplot(dataframe, column_name, title=None, annotations = False):
    '''
    Genera diagrama de barras para una variable categorica

    Parameters
    
    >> dataframe: Dataframe de Pandas
    >> column_name: Columna a diagramar
    >> title: Titulo personalizado del grafico. Por defecto, es el mismo nombre de la columna del dataframe
    >> annotations: Ver anotaciones sobre cada barra, por defecto Falso.
    
    Return
    
    Genera el diagrama
    '''
    plt.title(title or column_name)
    snsPlot = sns.countplot(
        dataframe[column_name], order=dataframe[column_name].value_counts().index)
    
    if (annotations):
        bbox = dict(boxstyle="round", fc="0.9", ec="red")
        for i in snsPlot.patches:
            plt.annotate(i.get_height(),
                         xy=(i.get_x() + i.get_width()/2, i.get_height()),
                         xytext=(0, 15), ha='center', va='center', textcoords='offset points',
                         bbox=bbox)

    plt.xlabel('')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45)

def plot_distribution(column_name, max_pivot, dataframe, title=None, min_pivot=0):
    '''
    Genera histograma y boxplot para una variable numérica

    Parameters
    
    >> column_name: Nombre de la columna
    >> max_pivot: Valor máximo de ajuste
    >> dataframe: Dataframe de Pandas
    >> min_pivot: Valor mínimo de ajuste, por defecto es 0 ('cero').

    Return
    
    Genera ambos gráficos.
    '''
    data_filtered = dataframe.query(f'{min_pivot} <= {column_name} <=  {max_pivot}')
    serie_filtered = data_filtered[column_name]
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.title(f'Histograma y Boxplot: {title or column_name}')
    sns.distplot(data_filtered[column_name])
    mean, median, mode = serie_filtered.mean(), serie_filtered.median(), serie_filtered.mode()[0]
    plt.axvline(mode, color='g')
    plt.axvline(median, color='b')
    plt.axvline(mean, color='r')    
    plt.legend(['Moda','Mediana','Media'])
    plt.xlabel('')
    plt.subplot(2, 1, 2)
    sns.boxplot(x=column_name, y='property_type', data=data_filtered, linewidth=0.8)
    plt.ylabel('Tipo de propiedad')
    plt.xlabel('')

def split_barplot(column_name, dataframe, partitions=2):
    '''
    Funcion para graficar diagramas de barras con numerosas categorias
    
    Parameters:
    
    -> column_name: Nombre de la columna
    -> dataframe: DataFrame de Pandas
    -> partitions= Cantidad de diagramas a graficar.
    
    Ejemplo:
    
    split_plot('l3', data)
    split_plot('l3', data, 4)
    '''
    counts = dataframe[column_name].value_counts()
    split  = np.array_split(counts, partitions)
    colors = ['#C14242','#BF7F3F','#26724C','#BFBF3F','#3F7FBF','#8C8C72','#6AD264']
    for i, dataSplit in enumerate(split):
        plt.figure(figsize=(14,2))
        etiquetas = dataSplit.keys()
        valores = dataSplit.values
        plt.title('{} - {} [{}/{}]'.format(etiquetas[0], etiquetas[-1], i+1, partitions))
        plt.bar(etiquetas, valores, color = colors)
        plt.xticks(rotation = 73.5)
        plt.tick_params('x', labelsize = 12)
        plt.show()

def plot_price_by_period(dataframe):
    '''
    Funcion para graficar un lineplot de Seaborn del precio por periodo
    
    Parameters:
    
    -> dataframe: DataFrame de Pandas, debe contener las columnas 'price', 'period' y 'property_type'.
    
    Ejemplo:
    plot_price_by_period(data_filtered)
    '''
    plt.figure(figsize=(15,5))
    plt.title('Evolucion Precio/Periodo')
    sns.lineplot(x='period', y='price', data=dataframe, hue='property_type')
    lg=plt.legend(loc='upper left', bbox_to_anchor=(-.1, 1.25), ncol=4, fontsize='small')
    lg.texts[0].set_text('Tipo de propiedad:')
    plt.tick_params(labelsize=12)
    plt.xlabel('Periodos')
    plt.ylabel('Precio')
    
def plot_heatmaps(dataframe):
    '''
    Funcion para graficar un HeatMap basado en las correlaciones de Pearson y Spearman.
    
    Parameters:
    -> dataframe: DataFrame de Pandas.
    
    Ejemplo:
    plot_heatmaps(data_filtered)
    '''
    plt.figure(figsize=(17,6))
    plt.subplot(1,2,1)
    plt.title('Pearson')
    sns.heatmap(dataframe.corr(), annot=True, cmap=plt.cm.summer, square=True, fmt='.2f')
    plt.xticks(rotation=45)
    plt.subplot(1,2,2)
    plt.title('Spearman')
    sns.heatmap(dataframe.corr(method='spearman'), annot=True, cmap=plt.cm.summer, square=True, fmt='.2f')
    plt.xticks(rotation=45)

def plot_rmse_curve(model_type, X_train, X_test, y_train, y_test, k_values):
    '''
    Grafica la curva de errores RMSE para KNN o Arboles de decision.
    
    Parameters:
    
    -> model_type: String "knn" o "tree"
    -> X_train: Conjunto de datos de entrenamiento
    -> y_train: Variable objetivo de entrenamiento
    -> X_test: Conjunto de datos de validacion
    -> y_test: Variable objetivo de validacion
    -> k_values: Array de numeros enteros.
    
    Returns:
    Curva de errores RMSE
    
    '''
    list_rmse_train = np.zeros(len(k_values))
    list_rmse_test = list_rmse_train.copy()
    for index, k in enumerate(k_values):
        model = DecisionTreeRegressor(max_depth=k, random_state=42)
        if (model_type=='knn'):
            model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        model.fit(X_train, y_train)
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        list_rmse_train[index]=rmse_train
        list_rmse_test[index]=rmse_test
    title='Curva de Errores: '
    title+='Tree' if model_type=='tree' else 'Knn'
    plt.title(title, fontdict=dict(fontsize= 12))
    plt.plot(k_values, list_rmse_train, 'bo-')
    plt.plot(k_values, list_rmse_test, 'ro-')
    plt.xlabel('Complejidad')
    plt.ylabel('RMSE')
    plt.legend(['Train', 'Test'])
    
def plot_validation_curve(model_type, X, y, k_values):
    '''
    Grafica la curva de validacion para KNN o Arboles de decision.
    
    Parameters:
    
    -> model_type: String "knn" o "tree"
    -> X: Dataframe con dimensiones de estudio
    -> y: Variable objetivo de entrenamiento.
    -> k_values: Array de numeros enteros.
    
    Returns:
    Curva de validacion
    '''
    validation_curve_model= validation_curve(DecisionTreeRegressor(), X, y, param_name="max_depth", param_range= k_values, cv=10, n_jobs=-1)
    if (model_type=='knn'):
        validation_curve_model= validation_curve(KNeighborsRegressor(weights='distance'), X, y, param_name="n_neighbors", param_range= k_values, cv=10, n_jobs=-1)
    train_score, val_score = validation_curve_model
    title='Curva de Validación: '
    title+='Tree' if model_type=='tree' else 'Knn'
    plt.title(title, fontdict=dict(fontsize= 12))
    plt.plot(k_values, np.median(train_score, 1), 'bo-', label='training score')
    plt.plot(k_values, np.median(val_score, 1), 'ro-', label='validation score')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlabel('degree')
    plt.ylabel('score')

def get_haversine_distance(lat1, lon1, lat2, lon2):
    '''
    Calcula la distancia entre dos ubicaciones
    
    Parameters:
    -> lat1: Latitud origen
    -> lon1: Longitud origen
    -> lat2: Latitud destino
    -> lon2: Latitud destino
    
    Returns:
    Retorna un valor escalar en Km correspondiente a la distancia entre estos dos puntos.
    '''
    EARTH_RATIO= 6471 
    rad_lat1, rad_lat2=np.radians(lat1), np.radians(lat2)
    delta_lat=np.radians(lat2-lat1)    
    delta_lon=np.radians(lon2-lon1)    
    cos_lat1,cos_lat2=np.cos(rad_lat1), np.cos(rad_lat2)
    data =(np.sin(delta_lat/2)**2)+cos_lat1*cos_lat2*(np.sin(delta_lon/2)**2)
    distance= 2*EARTH_RATIO*np.arcsin(np.sqrt(data))
    return distance

def get_nearest_apartments(dataframe, lat, lon, n=3):
    '''
    Funcion que retorna un dataframe con las n propiedades mas cercanas a una ubicacion dadas sus coordenadas.
    
    Parameters:
    -> dataframe: DataFrame de Pandas, debe contener las columnas 'lat' y 'lon'.
    -> lat: Latitud
    -> lon: Longitud
    -> n: Numero de propiedades a buscar, por defecto n=3
    
    Return:
    Retorna un dataframe filtrado.
    
    Ejemplo:
    get_nearest_apartments(data, -34.6037389, -58.3837591)
    '''
    local_df=dataframe.copy()
    local_df['distance (km)']=local_df[['lat','lon']].apply(lambda x:get_haversine_distance(x[0], x[1], lat, lon), axis=1)
    return local_df.nsmallest(n, 'distance (km)')

def plot_errors_distribution(models_name, models_object, X_train, y_train, X_test, y_test):
    '''
    Reporte del cálculo del RMSE para cada conjunto (train y test).
    Un gráfico de dispersión de  y real vs  y predicho  para el conjunto de test.
    El histograma de los errores ( y − y predicho ) para cada conjunto.
    
    Parameters:
    -> models_name: Lista de string para cada modelo en forma legible.
    -> models_object: Lista de objetos de tipo modelos de regresion.
    -> X_train: Conjunto de datos de entrenamiento
    -> y_train: Variable objetivo de entrenamiento
    -> X_test: Conjunto de datos de validacion
    -> y_test: Variable objetivo de validacion
    
    Return:
    Reporte
    
    '''
    for model_name, model_obj in zip(models_name, models_object):
        y_train_pred = model_obj.predict(X_train)
        y_test_pred = model_obj.predict(X_test)
        y_test = np.array(y_test).reshape(-1, 1)
        y_train = np.array(y_train).reshape(-1, 1)
        y_train_pred = np.array(y_train_pred).reshape(-1, 1)
        y_test_pred = np.array(y_test_pred).reshape(-1, 1)

        print(f'Modelo: {model_name}')

        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        print(f'Raíz del error cuadrático medio en Train: {rmse_train}')
        print(f'Raíz del error cuadrático medio en Test: {rmse_test}')

        plt.figure(figsize=(15, 4))

        plt.subplot(1, 2, 1)
        sns.distplot(y_train - y_train_pred, bins=20, label='train')
        sns.distplot(y_test - y_test_pred, bins=20, label='test')
        plt.xlabel('errores')
        plt.legend()

        ax = plt.subplot(1, 2, 2)
        ax.scatter(y_test, y_test_pred, s=2)

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes]
        ]

        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        plt.xlabel('y (test)')
        plt.ylabel('y_pred (test)')

        plt.tight_layout()
        plt.show()