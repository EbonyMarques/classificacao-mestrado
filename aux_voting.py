import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
import scikitplot as skplt
import time

def plot_roc_curve_1(y_test, y_prob, path):
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

    roc_curve_line4 = ax.get_lines()[2]
    label4 = roc_curve_line4.get_label()
    label4 = label4.split('(')[1]
    label4 = label4.replace('area ', 'Área')
    label4 = label4.replace(')','')
    label4 = label4.replace('=',':')
    label4 = 'Média micro — ' + label4
    roc_curve_line4.set_label(label4)

    roc_curve_line5 = ax.get_lines()[3]
    label5 = roc_curve_line5.get_label()
    label5 = label5.split('(')[1]
    label5 = label5.replace('area ', 'Área')
    label5 = label5.replace(')','')
    label5 = label5.replace('=',':')
    label5 = 'Média macro — ' + label5
    roc_curve_line5.set_label(label5)

    ax.legend()
    plt.savefig(path)
    plt.clf()

def plot_confusion_matrix(labels, y_test, y_pred, path):
    y_test_original = [labels[label] for label in y_test]
    y_pred_original = [labels[label] for label in y_pred]
    skplt.metrics.plot_confusion_matrix(y_test_original, y_pred_original, normalize=False, title=' ', figsize=(7,4))
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Predita')
    plt.savefig(path)
    plt.clf()

def voting_classification(data, labels, directory):
    clf1 = XGBClassifier(random_state=42)
    clf2 = SVC(probability=True, random_state=42)
    clf3 = LogisticRegression(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(data.drop('situacao_final_discente',axis=1), data['situacao_final_discente'], test_size=0.3, random_state=42)

    one_hot_tudo = ['area_cnpq_ciencias_agrarias','area_cnpq_ciencias_biologicas',
                                        'area_cnpq_ciencias_da_saude', 'area_cnpq_ciencias_exatas_e_da_terra',
                                        'turno_matutino', 'turno_noturno', 'turno_vespertino',
                                        'forma_ingresso_convenio', 'forma_ingresso_diplomado',
                                        'forma_ingresso_forca_de_lei', 'forma_ingresso_sisu',
                                        'forma_ingresso_transferencia_externa',
                                        'forma_ingresso_transferencia_interna','forma_ingresso_sub-judice']
    one_hot_aqui = []

    for i in one_hot_tudo:
        if i in X_train.columns:
            one_hot_aqui.append(i)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(one_hot_aqui, axis=1))
    X_test_scaled = scaler.transform(X_test.drop(one_hot_aqui, axis=1))
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.drop(one_hot_aqui, axis=1).columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.drop(one_hot_aqui, axis=1).columns)

    for i in one_hot_aqui:
        X_train_scaled_df[i] = X_train[i].reset_index(drop=True).copy()
        X_test_scaled_df[i] = X_test[i].reset_index(drop=True).copy()

    X_train = X_train_scaled_df.copy()
    X_test = X_test_scaled_df.copy()

    voting_classifier = VotingClassifier(
        estimators=[
            ('xgb', clf1),
            ('svc', clf2),
            ('log', clf3)
        ],
        voting='soft'  # Pode ser 'hard' ou 'soft'
    )

    start_time = time.time()
    voting_classifier.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    y_pred = voting_classifier.predict(X_test)
    y_pred = voting_classifier.predict(X_test)
    y_prob = voting_classifier.predict_proba(X_test)

    plot_roc_curve_1(y_test, y_prob, f'{directory}/voting/{directory}-voting-roc')

    plot_confusion_matrix(labels, y_test, y_pred, f'{directory}/voting/{directory}-voting-confusion-matrix')

    return voting_classifier, X_train, X_test, y_test, y_pred, y_prob