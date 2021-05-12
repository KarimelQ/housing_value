import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main():
    print("Hello World!")
    df = pd.read_csv('vf2020.txt', delimiter = "|")
    print(prepare_data(df))





def prepare_data(df):
    df = df[df['Code departement']==33]
    x = (df.isnull().sum() / df.shape[0] * 100).sort_values()
    col_to_drop = x[x.values >60].axes[0]
    df = df.drop(col_to_drop,axis=1)
    df['Date mutation'] = df['Date mutation'].apply(lambda x : x if pd.isna(x) else int(x[3:5])/12+int(x[8:10]))

    df['No disposition'] = df['No disposition'].apply(lambda x : 4 if(x>3) else x)

    nature_mutation_type = df['Nature mutation'].unique().tolist()
    df['Nature mutation'] = df["Nature mutation"].apply(lambda x: x if pd.isna(x) else  nature_mutation_type.index(x))


    df['Type de voie'] = df['Type de voie'].apply(lambda x : 'Autre' if pd.isna(x) else x)
    type_voie_counts = df['Type de voie'].value_counts() 
    col_to_keep = type_voie_counts[type_voie_counts.values > 3000].axes[0]
    df['Type de voie'] = df['Type de voie'].apply(lambda x : x if(x in col_to_keep) else 'Autre')

    df['Valeur fonciere'] = df['Valeur fonciere'].apply(lambda x : 0 if pd.isna(x) else  int(x.split(',')[0]))
    df['Valeur fonciere'] = df['Valeur fonciere'].apply(lambda x : df['Valeur fonciere'].mean() if x==0 else x)
    df = df[df['Valeur fonciere'] < 2000000]
    df = df.drop(['No voie','Type de voie','Code voie','Voie','Code postal','Commune','Code departement','Code commune','Section','No plan'],axis=1)
    df = df[df['Surface reelle bati'] < 300]
    df['Nature culture'] = df['Nature culture'].apply(lambda x : 'Autre' if pd.isna(x) else x)
    df['Surface terrain'] = df['Surface terrain'].apply(lambda x : df['Surface terrain'].mean() if pd.isna(x) else x)
    df['Code type local'] = df['Code type local'].apply(lambda x: str(x) )
    df = df.drop(['No disposition','Nature mutation','Type local','Date mutation','Nature culture'],axis=1)
    prix = df['Valeur fonciere']
    df = df.drop(['Valeur fonciere'],axis=1)
    cat_features = df.select_dtypes(include=[np.object])
    num_features = df.drop(cat_features.columns,axis=1)


    scaler = MinMaxScaler()
    scaler = scaler.fit(num_features)
    df = scaler.transform(num_features)
    return scaler



if __name__ == "__main__":
    main()


