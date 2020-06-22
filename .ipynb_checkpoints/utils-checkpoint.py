import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def generate_barplot(dataframe, column_name, title=None):
    '''
    Genera diagrama de barras para una variable categorica
    
    Parameters
    >> dataframe: Dataframe de Pandas
    >> column_name: Columna a diagramar
    >> title: Titulo personalizado del grafico. Por defecto, es el mismo nombre de la columna del dataframe
    Return
    No return. Genera el diagrama
    '''
    plt.title(title or column_name)
    sns.countplot(dataframe[column_name])
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
    filtered_df = dataframe.query(f'{min_pivot} <= {column_name} <=  {max_pivot}')
    serie_filtered = filtered_df[column_name]
    
    plt.figure(figsize=(15,5))
    plt.subplot(2,1,1)   
    plt.title(f'Histograma y Boxplot: {title or column_name}')
    sns.distplot(serie_filtered)
    plt.xlabel('')
    plt.subplot(2,1,2)
    sns.boxplot(serie_filtered, linewidth=0.6)
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
    Funcion para graficar un HeatMap basado en las correlaciones de Pearson y Spearman
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
    list_rmse_train = list_rmse_test = np.zeros(len(k_values))
    for index, k in enumerate(k_values):
        model = DecisionTreeRegressor(max_depth=k, random_state=42)
        if (model_type=='knn'):
            model = KNeighborsRegressor(n_neighbors=k)
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
    validation_curve_model= validation_curve(DecisionTreeRegressor(), X, y, param_name="max_depth", param_range= k_values, cv=10, n_jobs=-1)
    if (model_type=='knn'):
        validation_curve_model= validation_curve(KNeighborsRegressor(), X, y, param_name="n_neighbors", param_range= k_values, cv=10, n_jobs=-1)
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