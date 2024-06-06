import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
import scikitplot as skplt
import shap

def plot_roc_curve_1(y_test, y_prob, path):
    plt.clf()
    skplt.metrics.plot_roc_curve(y_test, y_prob, title='')
    ax = plt.gca()

    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')

    roc_curve_line1 = ax.get_lines()[0]
    label1 = roc_curve_line1.get_label()
    label1 = label1.split('(')[1]
    label1 = label1.replace('area ', 'Área')
    label1 = label1.replace(')','')
    label1 = 'Classe "1. Evasão" — ' + label1
    label1 = label1.replace('=',':')
    roc_curve_line1.set_label(label1)
    
    roc_curve_line2 = ax.get_lines()[1]
    label2 = roc_curve_line2.get_label()
    label2 = label2.split('(')[1]
    label2 = label2.replace('area ', 'Área')
    label2 = label2.replace(')','')
    label2 = 'Classe "2. Formação" — ' + label2
    label2 = label2.replace('=',':')
    roc_curve_line2.set_label(label2)

    roc_curve_line3 = ax.get_lines()[2]
    label3 = roc_curve_line3.get_label()
    label3 = label3.split('(')[1]
    label3 = label3.replace('area ', 'Área')
    label3 = label3.replace(')','')
    label3 = 'Média micro — ' + label3
    label3 = label3.replace('=',':')
    roc_curve_line3.set_label(label3)

    roc_curve_line4 = ax.get_lines()[3]
    label4 = roc_curve_line4.get_label()
    label4 = label4.split('(')[1]
    label4 = label4.replace('area ', 'Área')
    label4 = label4.replace(')','')
    label4 = label4.replace('=',':')
    label4 = 'Média macro — ' + label4
    roc_curve_line4.set_label(label4)

    ax.legend()

    plt.savefig(path)
    plt.clf()

def plot_roc_curve_4(y_test, y_prob, path):
    plt.clf()
    skplt.metrics.plot_roc_curve(y_test, y_prob, title='')
    ax = plt.gca()

    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')

    roc_curve_line1 = ax.get_lines()[0]
    label1 = roc_curve_line1.get_label()
    label1 = label1.split('(')[1]
    label1 = label1.replace('area ', 'Área')
    label1 = label1.replace(')','')
    label1 = 'Classe "1. Evasão" — ' + label1
    label1 = label1.replace('=',':')
    roc_curve_line1.set_label(label1)
    
    roc_curve_line2 = ax.get_lines()[1]
    label2 = roc_curve_line2.get_label()
    label2 = label2.split('(')[1]
    label2 = label2.replace('area ', 'Área')
    label2 = label2.replace(')','')
    label2 = 'Classe "2. Não evasão" — ' + label2
    label2 = label2.replace('=',':')
    roc_curve_line2.set_label(label2)

    roc_curve_line3 = ax.get_lines()[2]
    label3 = roc_curve_line3.get_label()
    label3 = label3.split('(')[1]
    label3 = label3.replace('area ', 'Área')
    label3 = label3.replace(')','')
    label3 = 'Média micro — ' + label3
    label3 = label3.replace('=',':')
    roc_curve_line3.set_label(label3)

    roc_curve_line4 = ax.get_lines()[3]
    label4 = roc_curve_line4.get_label()
    label4 = label4.split('(')[1]
    label4 = label4.replace('area ', 'Área')
    label4 = label4.replace(')','')
    label4 = label4.replace('=',':')
    label4 = 'Média macro — ' + label4
    roc_curve_line4.set_label(label4)

    ax.legend()

    plt.savefig(path)
    plt.clf()

def plot_roc_curve_6(y_test, y_prob, path):
    plt.clf()
    skplt.metrics.plot_roc_curve(y_test, y_prob, title='')
    ax = plt.gca()

    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')

    roc_curve_line1 = ax.get_lines()[0]
    label1 = roc_curve_line1.get_label()
    label1 = label1.split('(')[1]
    label1 = label1.replace('area ', 'Área')
    label1 = label1.replace(')','')
    label1 = 'Classe "1. Evasão" — ' + label1
    label1 = label1.replace('=',':')
    roc_curve_line1.set_label(label1)
    
    roc_curve_line2 = ax.get_lines()[1]
    label2 = roc_curve_line2.get_label()
    label2 = label2.split('(')[1]
    label2 = label2.replace('area ', 'Área')
    label2 = label2.replace(')','')
    label2 = 'Classe "2. Não evasão" — ' + label2
    label2 = label2.replace('=',':')
    roc_curve_line2.set_label(label2)

    roc_curve_line3 = ax.get_lines()[2]
    label3 = roc_curve_line3.get_label()
    label3 = label3.split('(')[1]
    label3 = label3.replace('area ', 'Área')
    label3 = label3.replace(')','')
    label3 = 'Média micro — ' + label3
    label3 = label3.replace('=',':')
    roc_curve_line3.set_label(label3)

    roc_curve_line4 = ax.get_lines()[3]
    label4 = roc_curve_line4.get_label()
    label4 = label4.split('(')[1]
    label4 = label4.replace('area ', 'Área')
    label4 = label4.replace(')','')
    label4 = label4.replace('=',':')
    label4 = 'Média macro — ' + label4
    roc_curve_line4.set_label(label4)

    ax.legend()

    plt.savefig(path)
    plt.clf()

def plot_confusion_matrix(y_test_original, y_pred_original, path):
    plt.clf()
    skplt.metrics.plot_confusion_matrix(y_test_original, y_pred_original, normalize=False, title=' ', figsize=(7,4))
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Predita')
    plt.savefig(path)
    plt.clf()

def explain(model, X_test, model_name, fold, directory):
    plt.clf()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.clf()
    shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=15, show=False)
    plt.savefig(f'{directory}/cv/{directory}-{model_name}-cv{fold}-importancia1')
    plt.clf()
    shap.summary_plot(shap_values, X_test, plot_type='dot', max_display=15, show=False)
    plt.savefig(f'{directory}/cv/{directory}-{model_name}-cv{fold}-importancia2')
    plt.clf()

def cross_validation_classification(data, target, models, labels, metrics, directory):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {metric: [] for metric in metrics}
    final_results = {}

    for model in models:
        fold = 0
        model_name = type(model).__name__
        final_results[model_name] = {}
        all_y_test, all_y_pred, all_y_prob, all_y_test_original, all_y_pred_original = [], [], [], [], []
        confusion_matrix_sum = None

        for metric in metrics:
            results[metric] = ([], [])

        one_hot_encoding_features = ['area_cnpq_ciencias_agrarias', 'area_cnpq_ciencias_biologicas',
                        'area_cnpq_ciencias_da_saude', 'area_cnpq_ciencias_exatas_e_da_terra',
                        'turno_matutino', 'turno_noturno', 'turno_vespertino',
                        'forma_ingresso_convenio', 'forma_ingresso_diplomado',
                        'forma_ingresso_forca_de_lei', 'forma_ingresso_sisu',
                        'forma_ingresso_transferencia_externa',
                        'forma_ingresso_transferencia_interna', 'forma_ingresso_sub-judice']
        
        for train_index, test_index in cv.split(data, data[target]):
            fold += 1
            print(f'Iniciando iteração {fold} do {model_name}...')
            data_train, data_test = data.iloc[train_index], data.iloc[test_index]
            X_train = data_train.drop(columns=[target]).reset_index(drop=True).copy()
            y_train = data_train[target].reset_index(drop=True).copy()
            X_test = data_test.drop(columns=[target]).reset_index(drop=True).copy()
            y_test = data_test[target].reset_index(drop=True).copy()
            one_hot_encoding_cv = []

            for i in one_hot_encoding_features:
                if i in X_train.columns:
                    one_hot_encoding_cv.append(i)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.drop(one_hot_encoding_cv, axis=1))
            X_test_scaled = scaler.transform(X_test.drop(one_hot_encoding_cv, axis=1))
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.drop(one_hot_encoding_cv, axis=1).columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.drop(one_hot_encoding_cv, axis=1).columns)

            for i in one_hot_encoding_cv:
                X_train_scaled_df[i] = X_train[i].reset_index(drop=True).copy()
                X_test_scaled_df[i] = X_test[i].reset_index(drop=True).copy()

            X_train = X_train_scaled_df.copy()
            X_test = X_test_scaled_df.copy()

            start_time = time.time()
            model.fit(X_train, y_train)
            end_time = time.time()
            elapsed_time = end_time - start_time

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            y_test_original = [labels[label] for label in y_test]
            y_pred_original = [labels[label] for label in y_pred]
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)
            all_y_test_original.extend(y_test_original)
            all_y_pred_original.extend(y_pred_original)

            if directory == '1':
                plot_roc_curve_1(y_test, y_prob, f'{directory}/cv/{directory}-{model_name}-cv{fold}-roc')
            elif directory == '4':
                plot_roc_curve_4(y_test, y_prob, f'{directory}/cv/{directory}-{model_name}-cv{fold}-roc')
            elif directory == '6':
                plot_roc_curve_6(y_test, y_prob, f'{directory}/cv/{directory}-{model_name}-cv{fold}-roc')

            for metric in metrics:
                if metric == 'accuracy':
                    acuracia = accuracy_score(y_test, y_pred)
                    results[metric][0].append(acuracia)
                elif metric == 'precision':
                    precisao_macro = precision_score(y_test, y_pred, average='macro')
                    precisao_classes = precision_score(y_test, y_pred, average=None)
                    results[metric][0].append(precisao_macro)
                    results[metric][1].append(precisao_classes)
                elif metric == 'recall':
                    recall_macro = recall_score(y_test, y_pred, average='macro')
                    recall_classes = recall_score(y_test, y_pred, average=None)
                    results[metric][0].append(recall_macro)
                    results[metric][1].append(recall_classes)
                elif metric == 'f1':
                    f1_macro = f1_score(y_test, y_pred, average='macro')
                    f1_classes = f1_score(y_test, y_pred, average=None)
                    results[metric][0].append(f1_macro)
                    results[metric][1].append(f1_classes)

            cm_fold = confusion_matrix(y_test_original, y_pred_original)

            plot_confusion_matrix(y_test_original, y_pred_original, f'{directory}/cv/{directory}-{model_name}-cv{fold}-matriz')

            if confusion_matrix_sum is None:
                confusion_matrix_sum = cm_fold
            else:
                confusion_matrix_sum += cm_fold

            explain(model, X_test, model_name, fold, directory)

            print(f'Iteração {fold} do {model_name} concluída.\n')

            if directory == '1':
                plot_roc_curve_1(y_test, y_prob, f'{directory}/cv/{directory}-{model_name}-resumo-roc')
            elif directory == '4':
                plot_roc_curve_4(y_test, y_prob, f'{directory}/cv/{directory}-{model_name}-resumo-roc')
            elif directory == '6':
                plot_roc_curve_6(y_test, y_prob, f'{directory}/cv/{directory}-{model_name}-resumo-roc')

        final_results[model_name]['elapsed_time'] = elapsed_time

        for metric in metrics:
            metric_mean = [None, None]
            mean = np.mean(results[metric][0])
            metric_mean[0] = mean
            if metric == "precision" or metric == "recall" or metric == "f1":
                values_0, values_1 = [], []
                for i in range(0, 5):
                    values_0.append(results[metric][1][i][0])
                    values_1.append(results[metric][1][i][1])
                value_0 = np.mean(values_0)
                value_1 = np.mean(values_1)
                metric_mean[1] = (value_0, value_1)
            final_results[model_name][metric] = metric_mean

        plot_confusion_matrix(all_y_test_original, all_y_pred_original, f'{directory}/cv/{directory}-{model_name}-resumo-matriz')

    return final_results