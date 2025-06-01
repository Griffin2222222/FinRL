from sklearn.preprocessing import RobustScaler

def normalize_features(df, columns):
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df