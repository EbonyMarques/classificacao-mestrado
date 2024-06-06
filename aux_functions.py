from unidecode import unidecode
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

def remove_acentos(texto):
    return unidecode(texto)

def minusculo(texto):
    texto = texto.replace(' ', '_')
    return str(texto).lower()

def load_and_preprocess_data_1():
   df = pd.read_parquet('../dados/extract_ebony_clusters.parquet')

   df = df.query('situacao_periodo_letivo != "SUSPENSO"').reset_index(drop=True).copy()
   df = df.query('ano_ingresso >= 2010').reset_index(drop=True).copy()
   df = df.groupby('id_discente').filter(lambda x: len(x) > 1).copy()
   df = df.sort_values(['id_discente', 'periodo_letivo'])
   
   df['situacao_final_discente'] = df.groupby('id_discente')['situacao_periodo_letivo'].transform('last').copy()
   df['periodo_final'] = df.groupby('id_discente')['duracao_vinculo'].transform('last').copy()
   
   df_final = df.query('(situacao_final_discente == "EVADIDO") | (situacao_final_discente == "FORMADO")').reset_index(drop=True).copy()
   df_final.replace({'situacao_final_discente':{'FORMADO':'formado','EVADIDO':'evadido'}},inplace=True)
   df_final = df_final.query('situacao_periodo_letivo != "EVADIDO"').copy()

   df_final = df_final.query('duracao_vinculo != periodo_final').reset_index(drop=True).copy()

   mapeamento_personalizado = {
    'evadido': 0,
    'formado': 1,
    }
   df_final['situacao_final_discente'] = df_final['situacao_final_discente'].map(mapeamento_personalizado)

   df_final['diferenca_conc_em_inicio_graduacao'] = df_final['ano_ingresso'].copy() - df_final['ano_conclusao_medio'].copy()
   df_final['diferenca_conc_em_inicio_graduacao'].fillna(df_final['diferenca_conc_em_inicio_graduacao'].mean(), inplace=True)
   
   df_final = df_final[['id_discente','duracao_vinculo','area_cnpq','diferenca_conc_em_inicio_graduacao','forma_ingresso','periodo_ingresso','turno','media_final','ch_total','ch_aprovacoes_acum','ch_reprovacoes_acum','ch_cancelamentos_acum','situacao_final_discente']].copy()
   df_final = df_final.query('media_final.notnull() and ch_total.notnull() and ch_aprovacoes_acum.notnull() and ch_reprovacoes_acum.notnull() and ch_cancelamentos_acum.notnull()').copy()

   df_final.columns = df_final.columns.str.lower()
   colunas_categoricas = df_final.select_dtypes(include=['object']).columns
   for coluna in colunas_categoricas:
      df_final[coluna] = df_final[coluna].apply(remove_acentos)
      df_final[coluna] = df_final[coluna].apply(minusculo)

   enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
   categoricas = ['area_cnpq','turno','forma_ingresso']
   data_enc = enc.fit_transform(df_final[categoricas])
   enc_df = pd.DataFrame(data_enc, columns=enc.get_feature_names_out())
   enc_df = pd.concat([enc_df.reset_index(drop=True),df_final.drop(columns=categoricas).reset_index(drop=True)],axis=1).copy()
   df_final = enc_df.copy()

   rus = RandomUnderSampler(random_state=42)
   X = df_final.drop(['situacao_final_discente'],axis=1).copy()
   y = df_final.situacao_final_discente.copy()
   X, y = rus.fit_resample(X, y)
   X['situacao_final_discente'] = y.copy()
   df_final = X.copy()

   print(df_final.situacao_final_discente.value_counts())
   
   return df_final

def load_and_preprocess_data_2():
   df = pd.read_parquet('../dados/extract_ebony_clusters.parquet')

   df = df.query('situacao_periodo_letivo != "SUSPENSO"').reset_index(drop=True).copy()
   df = df.query('ano_ingresso >= 2010').reset_index(drop=True).copy()
   df = df.groupby('id_discente').filter(lambda x: len(x) > 1).copy()
   df = df.sort_values(['id_discente', 'periodo_letivo'])
   
   df['situacao_final_discente'] = df.groupby('id_discente')['situacao_periodo_letivo'].transform('last').copy()
   df['periodo_final'] = df.groupby('id_discente')['duracao_vinculo'].transform('last').copy()
   
   df_final = df.query('(situacao_final_discente == "EVADIDO") | (situacao_final_discente == "FORMADO")').reset_index(drop=True).copy()
   df_final.replace({'situacao_final_discente':{'FORMADO':'formado','EVADIDO':'evadido'}},inplace=True)
   df_final = df_final.query('situacao_periodo_letivo != "EVADIDO"').copy()

   df_final['situacao_final_discente'] = df_final.apply(lambda row: 'formado' if (row['situacao_final_discente'] == 'formado') else
                                            'evadido_ate_quatro_anos' if ((row['periodo_final'] <= 9) and (row['situacao_final_discente'] == 'evadido')) else 
                                            'evadido_apos_quatro_anos' if ((row['periodo_final'] > 9) and (row['situacao_final_discente'] == 'evadido')) else
                                            row['situacao_final_discente'], axis=1)

   df_final = df_final.query('duracao_vinculo != periodo_final').reset_index(drop=True).copy()
   
   mapeamento_personalizado = {
    'evadido_ate_quatro_anos': 0,
    'evadido_apos_quatro_anos': 1,
    'formado': 2,
    }
   df_final['situacao_final_discente'] = df_final['situacao_final_discente'].map(mapeamento_personalizado)
   
   df_final['diferenca_conc_em_inicio_graduacao'] = df_final['ano_ingresso'].copy() - df_final['ano_conclusao_medio'].copy()
   df_final['diferenca_conc_em_inicio_graduacao'].fillna(df_final['diferenca_conc_em_inicio_graduacao'].mean(), inplace=True)
   
   df_final = df_final[['id_discente','duracao_vinculo','area_cnpq','diferenca_conc_em_inicio_graduacao','forma_ingresso','periodo_ingresso','turno','media_final','ch_total','ch_aprovacoes_acum','ch_reprovacoes_acum','ch_cancelamentos_acum','situacao_final_discente']].copy()
   df_final = df_final.query('media_final.notnull() and ch_total.notnull() and ch_aprovacoes_acum.notnull() and ch_reprovacoes_acum.notnull() and ch_cancelamentos_acum.notnull()').copy()

   df_final.columns = df_final.columns.str.lower()
   colunas_categoricas = df_final.select_dtypes(include=['object']).columns
   for coluna in colunas_categoricas:
      df_final[coluna] = df_final[coluna].apply(remove_acentos)
      df_final[coluna] = df_final[coluna].apply(minusculo)

   enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
   categoricas = ['area_cnpq','turno','forma_ingresso']
   data_enc = enc.fit_transform(df_final[categoricas])
   enc_df = pd.DataFrame(data_enc, columns=enc.get_feature_names_out())
   enc_df = pd.concat([enc_df.reset_index(drop=True),df_final.drop(columns=categoricas).reset_index(drop=True)],axis=1).copy()
   df_final = enc_df.copy()

   rus = RandomUnderSampler(random_state=42)
   X = df_final.drop(['situacao_final_discente'],axis=1).copy()
   y = df_final.situacao_final_discente.copy()
   X, y = rus.fit_resample(X, y)
   X['situacao_final_discente'] = y.copy()
   df_final = X.copy()

   print(df_final['situacao_final_discente'].value_counts())
   
   return df_final

def load_and_preprocess_data_3():
   df = pd.read_parquet('../dados/extract_ebony_clusters.parquet')

   df = df.query('situacao_periodo_letivo != "SUSPENSO"').reset_index(drop=True).copy()
   df = df.query('ano_ingresso >= 2010').reset_index(drop=True).copy()
   df = df.groupby('id_discente').filter(lambda x: len(x) > 1).copy()
   df = df.sort_values(['id_discente', 'periodo_letivo'])
   
   df['situacao_final_discente'] = df.groupby('id_discente')['situacao_periodo_letivo'].transform('last').copy()
   df['periodo_final'] = df.groupby('id_discente')['duracao_vinculo'].transform('last').copy()
   
   df_final = df.query('(situacao_final_discente == "EVADIDO") | (situacao_final_discente == "FORMADO")').reset_index(drop=True).copy()
   df_final.replace({'situacao_final_discente':{'FORMADO':'formado','EVADIDO':'evadido'}},inplace=True)
   df_final = df_final.query('situacao_periodo_letivo != "EVADIDO"').copy()

   df_final['situacao_final_discente'] = df_final.apply(lambda row: 'formado' if (row['situacao_final_discente'] == 'formado') else
                                            'evadido_ate_dois_anos' if ((row['periodo_final'] <= 5) and (row['situacao_final_discente'] == 'evadido')) else 
                                            'evadido_entre_dois_e_quatro_anos' if ((row['periodo_final'] > 5) and (row['periodo_final'] <= 9) and (row['situacao_final_discente'] == 'evadido')) else
                                            'evadido_apos_quatro_anos' if ((row['periodo_final'] > 9) and (row['situacao_final_discente'] == 'evadido')) else
                                            row['situacao_final_discente'], axis=1)

   df_final = df_final.query('duracao_vinculo != periodo_final').reset_index(drop=True).copy()
   
   mapeamento_personalizado = {
    'evadido_ate_dois_anos': 0,
    'evadido_entre_dois_e_quatro_anos': 1,
    'evadido_apos_quatro_anos': 2,
    'formado': 3,
    }
   df_final['situacao_final_discente'] = df_final['situacao_final_discente'].map(mapeamento_personalizado)
   
   df_final['diferenca_conc_em_inicio_graduacao'] = df_final['ano_ingresso'].copy() - df_final['ano_conclusao_medio'].copy()
   df_final['diferenca_conc_em_inicio_graduacao'].fillna(df_final['diferenca_conc_em_inicio_graduacao'].mean(), inplace=True)
   
   df_final = df_final[['id_discente','duracao_vinculo','area_cnpq','diferenca_conc_em_inicio_graduacao','forma_ingresso','periodo_ingresso','turno','media_final','ch_total','ch_aprovacoes_acum','ch_reprovacoes_acum','ch_cancelamentos_acum','situacao_final_discente']].copy()
   df_final = df_final.query('media_final.notnull() and ch_total.notnull() and ch_aprovacoes_acum.notnull() and ch_reprovacoes_acum.notnull() and ch_cancelamentos_acum.notnull()').copy()

   df_final.columns = df_final.columns.str.lower()
   colunas_categoricas = df_final.select_dtypes(include=['object']).columns
   for coluna in colunas_categoricas:
      df_final[coluna] = df_final[coluna].apply(remove_acentos)
      df_final[coluna] = df_final[coluna].apply(minusculo)

   enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
   categoricas = ['area_cnpq','turno','forma_ingresso']
   data_enc = enc.fit_transform(df_final[categoricas])
   enc_df = pd.DataFrame(data_enc, columns=enc.get_feature_names_out())
   enc_df = pd.concat([enc_df.reset_index(drop=True),df_final.drop(columns=categoricas).reset_index(drop=True)],axis=1).copy()
   df_final = enc_df.copy()

   rus = RandomUnderSampler(random_state=42)
   X = df_final.drop(['situacao_final_discente'],axis=1).copy()
   y = df_final.situacao_final_discente.copy()
   X, y = rus.fit_resample(X, y)
   X['situacao_final_discente'] = y.copy()
   df_final = X.copy()

   print(df_final.situacao_final_discente.value_counts())
   
   return df_final

def load_and_preprocess_data_4():
   select_periodo_final, select_duracao_vinculo = 5, 4

   df = pd.read_parquet('../dados/extract_ebony_clusters.parquet')

   df = df.query('situacao_periodo_letivo != "SUSPENSO"').reset_index(drop=True).copy()
   df = df.query('ano_ingresso >= 2010').reset_index(drop=True).copy()
   df = df.groupby('id_discente').filter(lambda x: len(x) > 1).copy()
   df = df.sort_values(['id_discente', 'periodo_letivo'])

   df['situacao_final_discente'] = df.groupby('id_discente')['situacao_periodo_letivo'].transform('last').copy()
   df['periodo_final'] = df.groupby('id_discente')['duracao_vinculo'].transform('last').copy()

   df1 = df.query('(situacao_final_discente == "EVADIDO" & periodo_final <= @select_periodo_final)').reset_index(drop=True).copy()
   df2 = df.query('(periodo_final > @select_periodo_final)').reset_index(drop=True).copy()
   df1 = df1.query('duracao_vinculo <= @select_duracao_vinculo')
   df2 = df2.query('duracao_vinculo <= @select_duracao_vinculo')
   df1.replace({'situacao_final_discente':{'CURSANDO':'nao_evadido','FORMADO':'nao_evadido','EVADIDO':'evadido'}},inplace=True) # apenas os evadidos no semestre de select_periodo_final interessam aqui; daí pego os dados do período anterior, pois o período final não tem dados
   df2.replace({'situacao_final_discente':{'CURSANDO':'nao_evadido','FORMADO':'nao_evadido','EVADIDO':'nao_evadido'}},inplace=True) # os evadidos após o semestre de select_periodo_final não interessam aqui
   
   df_final = pd.concat([df1.reset_index(drop=True),df2.reset_index(drop=True)],axis=0)

   df_final = df_final.query('situacao_periodo_letivo != "EVADIDO"').copy()

   mapeamento_personalizado = {
    'evadido': 0,
    'nao_evadido': 1,
    }
   df_final['situacao_final_discente'] = df_final['situacao_final_discente'].map(mapeamento_personalizado)
   
   df_final['diferenca_conc_em_inicio_graduacao'] = df_final['ano_ingresso'].copy() - df_final['ano_conclusao_medio'].copy()
   df_final['diferenca_conc_em_inicio_graduacao'].fillna(df_final['diferenca_conc_em_inicio_graduacao'].mean(), inplace=True)
   
   df_final = df_final[['id_discente','duracao_vinculo','area_cnpq','diferenca_conc_em_inicio_graduacao','forma_ingresso','periodo_ingresso','turno','media_final','ch_total','ch_aprovacoes_acum','ch_reprovacoes_acum','ch_cancelamentos_acum','situacao_final_discente']].copy()
   df_final = df_final.query('media_final.notnull() and ch_total.notnull() and ch_aprovacoes_acum.notnull() and ch_reprovacoes_acum.notnull() and ch_cancelamentos_acum.notnull()').copy()

   df_final.columns = df_final.columns.str.lower()
   colunas_categoricas = df_final.select_dtypes(include=['object']).columns
   for coluna in colunas_categoricas:
      df_final[coluna] = df_final[coluna].apply(remove_acentos)
      df_final[coluna] = df_final[coluna].apply(minusculo)

   enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
   categoricas = ['area_cnpq','turno','forma_ingresso']
   data_enc = enc.fit_transform(df_final[categoricas])
   enc_df = pd.DataFrame(data_enc, columns=enc.get_feature_names_out())
   enc_df = pd.concat([enc_df.reset_index(drop=True),df_final.drop(columns=categoricas).reset_index(drop=True)],axis=1).copy()
   df_final = enc_df.copy()

   rus = RandomUnderSampler(random_state=42)
   X = df_final.drop(['situacao_final_discente'],axis=1).copy()
   y = df_final.situacao_final_discente.copy()
   X, y = rus.fit_resample(X, y)
   X['situacao_final_discente'] = y.copy()
   df_final = X.copy()

   print(df_final.situacao_final_discente.value_counts())
   
   return df_final

def load_and_preprocess_data_5():
   select_periodo_final, select_duracao_vinculo = 5, 4

   df = pd.read_parquet('../dados/extract_ebony_clusters.parquet')

   df = df.query('situacao_periodo_letivo != "SUSPENSO"').reset_index(drop=True).copy()
   df = df.query('ano_ingresso >= 2010').reset_index(drop=True).copy()
   df = df.groupby('id_discente').filter(lambda x: len(x) > 1).copy()
   df = df.sort_values(['id_discente', 'periodo_letivo'])

   df['situacao_final_discente'] = df.groupby('id_discente')['situacao_periodo_letivo'].transform('last').copy()
   df['periodo_final'] = df.groupby('id_discente')['duracao_vinculo'].transform('last').copy()

   df1 = df.query('(situacao_final_discente == "EVADIDO" & periodo_final <= @select_periodo_final)').reset_index(drop=True).copy()
   df2 = df.query('(periodo_final > @select_periodo_final)').reset_index(drop=True).copy()
   df1 = df1.query('duracao_vinculo <= @select_duracao_vinculo')
   df2 = df2.query('duracao_vinculo <= @select_duracao_vinculo')
   df1.replace({'situacao_final_discente':{'CURSANDO':'nao_evadido','FORMADO':'nao_evadido','EVADIDO':'evadido'}},inplace=True) # apenas os evadidos no semestre de select_periodo_final interessam aqui; daí pego os dados do período anterior, pois o período final não tem dados
   df2.replace({'situacao_final_discente':{'CURSANDO':'nao_evadido','FORMADO':'nao_evadido','EVADIDO':'nao_evadido'}},inplace=True) # os evadidos após o semestre de select_periodo_final não interessam aqui
   
   df_final = pd.concat([df1.reset_index(drop=True),df2.reset_index(drop=True)],axis=0)

   df_final = df_final.query('situacao_periodo_letivo != "EVADIDO"').copy()

   df_final['situacao_final_discente'] = df_final.apply(lambda row: 'nao_evadido_ate_o_segundo_ano' if (row['situacao_final_discente'] == 'nao_evadido') else
                                            'evadido_no_primeiro_ano' if ((row['periodo_final'] <= 3) and (row['situacao_final_discente'] == 'evadido')) else 
                                            'evadido_no_segundo_ano' if ((row['periodo_final'] > 3) and (row['situacao_final_discente'] == 'evadido')) else
                                            row['situacao_final_discente'], axis=1)
   
   mapeamento_personalizado = {
    'evadido_no_primeiro_ano': 0,
    'evadido_no_segundo_ano': 1,
    'nao_evadido_ate_o_segundo_ano': 2,
    }
   df_final['situacao_final_discente'] = df_final['situacao_final_discente'].map(mapeamento_personalizado)
   
   df_final['diferenca_conc_em_inicio_graduacao'] = df_final['ano_ingresso'].copy() - df_final['ano_conclusao_medio'].copy()
   df_final['diferenca_conc_em_inicio_graduacao'].fillna(df_final['diferenca_conc_em_inicio_graduacao'].mean(), inplace=True)
   
   df_final = df_final[['id_discente','duracao_vinculo','area_cnpq','diferenca_conc_em_inicio_graduacao','forma_ingresso','periodo_ingresso','turno','media_final','ch_total','ch_aprovacoes_acum','ch_reprovacoes_acum','ch_cancelamentos_acum','situacao_final_discente']].copy()
   df_final = df_final.query('media_final.notnull() and ch_total.notnull() and ch_aprovacoes_acum.notnull() and ch_reprovacoes_acum.notnull() and ch_cancelamentos_acum.notnull()').copy()

   df_final.columns = df_final.columns.str.lower()
   colunas_categoricas = df_final.select_dtypes(include=['object']).columns
   for coluna in colunas_categoricas:
      df_final[coluna] = df_final[coluna].apply(remove_acentos)
      df_final[coluna] = df_final[coluna].apply(minusculo)

   enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
   categoricas = ['area_cnpq','turno','forma_ingresso']
   data_enc = enc.fit_transform(df_final[categoricas])
   enc_df = pd.DataFrame(data_enc, columns=enc.get_feature_names_out())
   enc_df = pd.concat([enc_df.reset_index(drop=True),df_final.drop(columns=categoricas).reset_index(drop=True)],axis=1).copy()
   df_final = enc_df.copy()

   rus = RandomUnderSampler(random_state=42)
   X = df_final.drop(['situacao_final_discente'],axis=1).copy()
   y = df_final.situacao_final_discente.copy()
   X, y = rus.fit_resample(X, y)
   X['situacao_final_discente'] = y.copy()
   df_final = X.copy()

   print(df_final.situacao_final_discente.value_counts())
   
   return df_final

def load_and_preprocess_data_6():
    select_periodo_final, select_duracao_vinculo = 2, 1

    df = pd.read_parquet('../dados/extract_ebony_clusters.parquet')

    df = df.query('situacao_periodo_letivo != "SUSPENSO"').reset_index(drop=True).copy()
    df = df.query('ano_ingresso >= 2010').reset_index(drop=True).copy()
    df = df.groupby('id_discente').filter(lambda x: len(x) > 1).copy()
    df = df.sort_values(['id_discente', 'periodo_letivo'])
    
    df['situacao_final_discente'] = df.groupby('id_discente')['situacao_periodo_letivo'].transform('last')
    df['periodo_final'] = df.groupby('id_discente')['duracao_vinculo'].transform('last')

    df1 = df.query('(situacao_final_discente == "EVADIDO" & periodo_final == @select_periodo_final)').reset_index(drop=True).copy()
    df2 = df.query('(periodo_final > @select_periodo_final)').reset_index(drop=True).copy()
    df1 = df1.query('duracao_vinculo == @select_duracao_vinculo')
    df2 = df2.query('duracao_vinculo == @select_duracao_vinculo')
    df1.replace({'situacao_final_discente':{'CURSANDO':'nao_evadido','FORMADO':'nao_evadido','EVADIDO':'evadido'}},inplace=True) # apenas os evadidos no semestre de select_periodo_final interessam aqui; daí pego os dados do período anterior, pois o período final não tem dados
    df2.replace({'situacao_final_discente':{'CURSANDO':'nao_evadido','FORMADO':'nao_evadido','EVADIDO':'nao_evadido'}},inplace=True) # os evadidos após o semestre de select_periodo_final não interessam aqui
    df_final = pd.concat([df1.reset_index(drop=True),df2.reset_index(drop=True)],axis=0)
    df_final.drop_duplicates('id_discente',inplace=True)
    df_final = df_final.query('situacao_periodo_letivo != "EVADIDO"').copy()

    mapeamento_personalizado = {
    'evadido': 0,
    'nao_evadido': 1,
    }

    df_final['situacao_final_discente'] = df_final['situacao_final_discente'].map(mapeamento_personalizado)

    df_final['diferenca_conc_em_inicio_graduacao'] = df_final['ano_ingresso'] - df_final['ano_conclusao_medio']
    df_final['diferenca_conc_em_inicio_graduacao'].fillna(df_final['diferenca_conc_em_inicio_graduacao'].mean(), inplace=True)
    df_final = df_final[['id_discente','area_cnpq','diferenca_conc_em_inicio_graduacao','forma_ingresso','periodo_ingresso','turno','media_final','ch_total','ch_aprovacoes','ch_reprovacoes','ch_cancelamentos','situacao_final_discente']]
    
    df_final.columns = df_final.columns.str.lower()
    colunas_categoricas = df_final.select_dtypes(include=['object']).columns
    for coluna in colunas_categoricas:
        df_final[coluna] = df_final[coluna].apply(remove_acentos)
        df_final[coluna] = df_final[coluna].apply(minusculo)

    rus = RandomUnderSampler(random_state=42)
    X = df_final.drop(['situacao_final_discente'],axis=1)
    y = df_final.situacao_final_discente
    X, y = rus.fit_resample(X, y)
    X['situacao_final_discente'] = y
    df_final = X.copy()

    print(df_final.situacao_final_discente.value_counts())
    
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    categoricas = ['area_cnpq','turno','forma_ingresso']
    data_enc = enc.fit_transform(X[categoricas])
    enc_df = pd.DataFrame(data_enc, columns=enc.get_feature_names_out())
    enc_df = pd.concat([enc_df.reset_index(drop=True),df_final.drop(columns=categoricas).reset_index(drop=True)],axis=1)
    df_final = enc_df.copy()
    df_final['situacao_final_discente'] = y
    
    return df_final