# scikit-learn == 1.3.0
# matplotlib == 3.7.2
# numpy == 1.24.4
# pandas == 2.0.3

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io.arff import loadarff
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)




# função para criar os diretórios
def creat_dir(model_name):
    if not 'log_metrics' in os.listdir():
        os.mkdir('log_metrics')
    if not 'imgs' in os.listdir():
        os.mkdir('imgs')
    if not model_name in os.listdir('./imgs'):
        os.mkdir('./imgs/'+model_name)




# função para adicionar as métricas de um fold em um dicionário
def dic_par_metrics(y_test, y_onehot_test, y_pred, y_proba, grid_search_cv):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
    precision_micro = metrics.precision_score(y_test, y_pred, average='micro')
    precision_macro = metrics.precision_score(y_test, y_pred, average='macro')
    precision_weighted = metrics.precision_score(y_test, y_pred, average='weighted')
    recall_micro = metrics.recall_score(y_test, y_pred, average='micro')
    recall_macro = metrics.recall_score(y_test, y_pred, average='macro')
    recall_weighted = metrics.recall_score(y_test, y_pred, average='weighted')
    f1_micro = metrics.f1_score(y_test, y_pred, average='micro')
    f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
    f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    roc_auc_ovr_micro = metrics.roc_auc_score(y_onehot_test, y_proba, multi_class='ovr', average='micro')
    roc_auc_ovo_micro = metrics.roc_auc_score(y_onehot_test, y_proba, multi_class='ovo', average='micro')
    roc_auc_ovr_macro = metrics.roc_auc_score(y_onehot_test, y_proba, multi_class='ovr', average='macro')
    roc_auc_ovo_macro = metrics.roc_auc_score(y_onehot_test, y_proba, multi_class='ovo', average='macro')
    roc_auc_ovr_weighted = metrics.roc_auc_score(y_onehot_test, y_proba, multi_class='ovr', average='weighted')
    roc_auc_ovo_weighted = metrics.roc_auc_score(y_onehot_test, y_proba, multi_class='ovo', average='weighted')
    dicMetricas = {
        "parameters": grid_search_cv.best_params_,
        "metrics":{
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision_micro": precision_micro,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_micro": recall_micro,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "roc_auc_ovr_micro": roc_auc_ovr_micro,
            "roc_auc_ovo_micro": roc_auc_ovo_micro,
            "roc_auc_ovr_macro": roc_auc_ovr_macro,
            "roc_auc_ovo_macro": roc_auc_ovo_macro,
            "roc_auc_ovr_weighted": roc_auc_ovr_weighted,
            "roc_auc_ovo_weighted": roc_auc_ovo_weighted
        }
    }
    return dicMetricas




# função para calcular a média e o desvio padrão das métricas de cada fold, além de plotar os boxplots
def calc_mean_std(dic_json, model_name):
    key = list(dic_json.keys())[0]
    dic_mean = {}
    dic_std = {}
    for j in dic_json[key]['metrics'].keys():
        metrics_list = []
        for i in dic_json.keys():
            metrics_list.append(dic_json[i]['metrics'][j])
        mean = np.mean(metrics_list)
        std = np.std(metrics_list)
        dic_mean.update({j: mean})
        dic_std.update({j: std})
        plt.close('all')
        plt.boxplot(metrics_list, labels=[model_name])
        plt.ylabel(j)
        plt.savefig('./imgs/'+model_name+'/boxplot_'+j+'.png')
        plt.close('all')
    dic_json.update({"mean": dic_mean})
    dic_json.update({"std": dic_std})
    return dic_json




def plot_matriz_confusao_one_vs_one(y_test, y_pred, model_name, fold_i, flag_normalizado="true"):
    """Plota a matriz de confusao comparando cada classe entre si, mostrando as predicoes contra as
    classes verdadeiras.

    Args:
        y_test: serie com os valores verdadeiros
        y_pred: serie com os valores preditos
        flag_normalizado (optional): flag indicando se os valores da matriz de confusao devem ser
        normalizados. Se devem ser normalizados pel. Defaults to None.
    """
    class_names = np.unique(y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    cm_plot = metrics.ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap="YlOrRd", normalize=flag_normalizado, ax=ax)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_title(f"Matriz de Confusao - Modelo {model_name} - Fold {fold_i}")

    cm_plot.figure_.savefig(
        f"imgs/{model_name}/cm_normalizado_{flag_normalizado}_{model_name}_fold_{fold_i}.png", 
        dpi=300
    )
    plt.close()
    
    return cm_plot




def inicializa_roc_one_class_vs_rest_kfold(y_onehot_test, y_pred_score, i, class_id, ax, tprs, aucs, mean_fpr):
    viz = metrics.RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_pred_score[:, class_id],
        name=f"ROC OvR Class_{class_id+1} fold {i}",
        alpha=0.3,
        lw=1,
        ax=ax,
        plot_chance_level=(i==9),
    )

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)




def gera_roc_one_class_vs_rest_kfold(tprs, aucs, mean_fpr, plot, model_name, class_id):
    fig = plot[0]
    ax = plot[1]
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Curva ROC media com desvio padrao\nROC OvR Class_{class_id+1}",
    )
    ax.axis("square")
    ax.legend(loc="lower right")

    fig.savefig(f"imgs/{model_name}/roc_{model_name}_class_{class_id+1}.png", dpi=300)
    plt.close()




def inicializa_roc_micro_average_kfold(y_onehot_test, y_pred_score, i, ax, tprs, aucs, mean_fpr):
    viz = metrics.RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_pred_score.ravel(),
        name=f"micro-average OvR fold {i}",
        alpha=0.3,
        lw=1,
        ax=ax,
        plot_chance_level=(i==9)
    )

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)




def gera_roc_micro_average_kfold(tprs, aucs, mean_fpr, fig, ax, model_name):
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Curva ROC media com desvio padrao\nROC micro-average",
    )
    ax.axis("square")
    ax.legend(loc="lower right")

    fig.savefig(f"imgs/{model_name}/roc_micro_average_{model_name}.png", dpi=300)
    plt.close()




def stratified_k_fold_grid_search_cv(model, params:dict, X, y, model_name):
    """Realiza o treino e validacao do modelo utilizando StratifiedKFold com GridSearchCV para
    busca dos melhores hiperparametros (tendo assim um nested cross-validation).

    Args:
        model: modelo que se deseja treinar e validar
        params (dict): dicionario com os parametros e seus respectivos valores possiveis para o 
        grid search
        X: features do modelo
        y: coluna de target do modelo
    """
    
    startTime = time.time()

    # criando os diretórios necessários para salvar as métricas e os plots
    creat_dir(model_name)
    
    print(f"Modelo: {model_name}")
    dic_json = {}

    n_splits_k_fold = 10
    n_splits_grid_search = 5

    matriz_tprs = [[] for i in range(9)]
    matriz_aucs = [[] for i in range(9)]
    mean_fpr = np.linspace(0, 1, 100)
    lista_plots = [plt.subplots(figsize=(6, 6)) for i in range(9)]

    tprs_micro_average = []
    aucs_micro_average = []
    fig_micro_average, ax_micro_average = plt.subplots(figsize=(6, 6))

    skf = StratifiedKFold(n_splits=n_splits_k_fold, random_state=4, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"## INICIO FOLD {i} ##")
        # separa os dados de treino e teste
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        # normaliza os dados
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # binariza a coluna target para uso no plot roc e metrica roc_auc_score
        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test)
        # prepara o y_train
        y_train = y_train.values.ravel()
        # inicializa e roda o grid search
        grid_search_cv = GridSearchCV(estimator=model,param_grid=params,scoring="f1_weighted",cv=n_splits_grid_search,verbose=3)
        grid_search_cv.fit(X_train, y_train)
        # com os melhores parametros encontrados, realiza a predicao no fold de teste e calcula a 
        # metrica de avaliacao
        y_pred = grid_search_cv.predict(X_test)
        if model_name != 'Perceptron':
            y_proba = grid_search_cv.predict_proba(X_test)
        else:
            y_proba = grid_search_cv.decision_function(X_test)
        # armazenando as metricas e os parametros em um dicionario
        dic_fold = dic_par_metrics(y_test, y_onehot_test, y_pred, y_proba, grid_search_cv)
        dic_json.update({"fold "+str(i): dic_fold})
        # gera a matriz de confusao com dados normalizados e nao normalizados e salva ambas versoes
        plot_matriz_confusao_one_vs_one(y_test, y_pred, model_name, i, flag_normalizado="true")
        plot_matriz_confusao_one_vs_one(y_test, y_pred, model_name, i, flag_normalizado=None)
        # inicializa a curva roc para cada classe neste fold
        for class_id, plot, tprs, aucs in zip(range(9), lista_plots, matriz_tprs, matriz_aucs):
            inicializa_roc_one_class_vs_rest_kfold(
                y_onehot_test, y_proba, i, class_id, plot[1], tprs, aucs, mean_fpr
            )
        # inicializa a curva roc micro average neste fold
        inicializa_roc_micro_average_kfold(
            y_onehot_test, y_proba, i, ax_micro_average, tprs_micro_average, 
            aucs_micro_average, mean_fpr
        )
        print(f"## FINAL FOLD {i} ##\n", dic_fold)
    
    endTime = time.time()
    # inserindo a média dos folds no dicionário
    dic_json = calc_mean_std(dic_json, model_name)
    # inserindo o tempo de execução
    dic_json.update({"time": endTime-startTime})
    # salvando o dicionário no formato json
    objOpen = open(f'./log_metrics/{model_name}.json', 'w')
    objOpen.write(json.dumps(dic_json, indent=4))
    objOpen.close()

    # gera o plot da curva roc para cada classe, juntando todos os folds e realizando uma media
    for class_id, plot, tprs, aucs in zip(range(9), lista_plots, matriz_tprs, matriz_aucs):
        gera_roc_one_class_vs_rest_kfold(tprs, aucs, mean_fpr, plot, model_name, class_id)
    # gera o plot da curva roc micro average, juntando todos os folds e realizando uma media 
    gera_roc_micro_average_kfold(
        tprs_micro_average, aucs_micro_average, mean_fpr, fig_micro_average, ax_micro_average, 
        model_name
    )




# importando o dataset
raw_data = loadarff('data/dataset.arff')
df_data = pd.DataFrame(raw_data[0])

# o alterando a coluna target (bytes para string)
df_data["target"] = df_data["target"].str.decode("utf-8")



# verificando o número de linhas e colunas do dataset
print("\nShape dataset:", df_data.shape)

# verificando o menor valor das colunas de características
print("\nValor mínimo das features:", df_data.iloc[:,1:-1].min().min())

# verificando o maior valor das colunas de características
print("\nValor máximo das features:", df_data.iloc[:,1:-1].max().max())

# verificando o número de instâncias para cada classe
print('\nNúmero de Instâncias por Classe:\n', df_data.groupby('target').count()['id'])

# verificando o percentual de instâncias para cada classe
print('\nPercentual de Instâncias por Classe:\n', (df_data.groupby('target').count()['id']/df_data.shape[0])*100)

# verificando se existem valores NaN
print('\nNúmero de Valores NaN:', df_data.isnull().sum().sum())



# separando as entradas e saídas desejadas
X = df_data.drop(columns=["id", "target"])
y = df_data[["target"]]


# o usuário irá inserir o nº referente ao modelo que deseja treinar
moedel_select = input("""
Insira o número referente ao modelo que deseja treinar:
1 - kNN
2 - Decision Tree
3 - Random Forest
4 - Naive Bayes 
5 - Logistic Regression 
6 - Perceptron
7 - MLP
8 - SVM
""")



## KNN
if moedel_select[0] == '1':
    # define os parametros e seus respectivos valores a serem testados no grid search
    params = {
        "n_neighbors": [5, 10, 15]
    }
    # define o modelo
    model = KNeighborsClassifier(n_jobs=1)
    # chama a funcao que roda o stratified k fold e valida o modelo realizando a busca por hiper 
    # parametros com grid search cv, fazendo assim um nested cross-validation
    stratified_k_fold_grid_search_cv(model, params, X, y, "KNN")



## DECISION TREE
if moedel_select[0] == '2':
    # define os parametros e seus respectivos valores a serem testados no grid search
    params = {
        "criterion": ["gini", "entropy", "log_loss"],
        "class_weight": ["balanced", None],
        "max_depth": [None, 50, 100, 200, 500, 1000]
    }
    # define o modelo
    model = DecisionTreeClassifier(random_state=4)
    # chama a funcao que roda o stratified k fold e valida o modelo realizando a busca por hiper 
    # parametros com grid search cv, fazendo assim um nested cross-validation
    stratified_k_fold_grid_search_cv(model, params, X, y, "DecisionTree")



## RANDOM FOREST
if moedel_select[0] == '3':
    # define os parametros e seus respectivos valores a serem testados no grid search
    params = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 400, 800],
        "class_weight": ["balanced_subsample", None]
    }
    # define o modelo
    model = RandomForestClassifier(random_state=4)
    # chama a funcao que roda o stratified k fold e valida o modelo realizando a busca por hiper 
    # parametros com grid search cv, fazendo assim um nested cross-validation
    stratified_k_fold_grid_search_cv(model, params, X, y, "RandomForest")



## NAIVE BAYES (GAUSSIAN)
if moedel_select[0] == '4':
    # define os parametros e seus respectivos valores a serem testados no grid search
    params = {
    }
    # define o modelo
    model = GaussianNB()
    # chama a funcao que roda o stratified k fold e valida o modelo realizando a busca por hiper 
    # parametros com grid search cv, fazendo assim um nested cross-validation
    stratified_k_fold_grid_search_cv(model, params, X, y, "NaiveBayes")



## LOGISTIC REGRSSION
if moedel_select[0] == '5':
    # define os parametros e seus respectivos valores a serem testados no grid search
    params = {
        "penalty": [None, "l2"],
        "class_weight": ["balanced", None],
        "max_iter": [100, 300, 500, 700, 1000]
    }
    # define o modelo
    model = LogisticRegression(random_state=4)
    # chama a funcao que roda o stratified k fold e valida o modelo realizando a busca por hiper 
    # parametros com grid search cv, fazendo assim um nested cross-validation
    stratified_k_fold_grid_search_cv(model, params, X, y, "LogisticRegression")



## PERCEPTRON
if moedel_select[0] == '6':
    # define os parametros e seus respectivos valores a serem testados no grid search
    params = {
        "penalty": [None, "l2"],
        "class_weight": ["balanced", None],
        "max_iter": [100, 300, 500, 700, 1000]
    }
    # define o modelo
    model = Perceptron(random_state=4)
    # chama a funcao que roda o stratified k fold e valida o modelo realizando a busca por hiper 
    # parametros com grid search cv, fazendo assim um nested cross-validation
    stratified_k_fold_grid_search_cv(model, params, X, y, "Perceptron")



## MLP
if moedel_select[0] == '7':
    # define os parametros e seus respectivos valores a serem testados no grid search
    params = {
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [200, 500, 700],
        "hidden_layer_sizes": [20, 50, 100]
    }
    # define o modelo
    model = MLPClassifier(random_state=4)
    # chama a funcao que roda o stratified k fold e valida o modelo realizando a busca por hiper 
    # parametros com grid search cv, fazendo assim um nested cross-validation
    stratified_k_fold_grid_search_cv(model, params, X, y, "MLPClassifier")



## SVM
if moedel_select[0] == '8':
    # define os parametros e seus respectivos valores a serem testados no grid search
    params = {
        "C": [1],
        "kernel": ["rbf"]
    }
    # define o modelo
    model = SVC(random_state=4, probability=True)
    # chama a funcao que roda o stratified k fold e valida o modelo realizando a busca por hiper 
    # parametros com grid search cv, fazendo assim um nested cross-validation
    stratified_k_fold_grid_search_cv(model, params, X, y, "SVM")
