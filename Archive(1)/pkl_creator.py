
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
games_df = pd.read_csv('games_table.csv', sep=',')
games_df.drop(['winner_id', 'p1_id', 'p2_id', 'diff_win', 'points_diff', 'p1_sets', 'p2_sets', 'Unnamed: 0', 'match_id',
               'match_type', 'date', 'p1_name', 'p2_name', 'p1_club', 'p2_club', 'p1_points_gained', 'p2_points_gained'], axis=1, inplace=True)

games_df.fillna(0, inplace=True)
games_df.info()
games_df['is_p1_win'] = games_df['p1_win']
games_df.drop(['p1_win', 'p2_win'], axis=1, inplace=True)
rf_df = games_df.copy()

rf_df['diff'] = rf_df['p1_rank'] - rf_df['p2_rank']
rf_df
rf_df = rf_df[((rf_df['diff'] >= -80) |
               ((rf_df['diff'] < -80) & rf_df['is_p1_win'] == 0))]
rf_df['is_p1_win'].value_counts()
r_X = rf_df.drop('is_p1_win', axis=1)
r_y = rf_df['is_p1_win']

temp_x = r_X.copy()
temp_x['diff'] = temp_x['p1_rank'] - temp_x['p2_rank']
temp_x = temp_x[['diff', 'p1_form', 'p2_form', 'p1_prevwins', 'p2_prevwins']]
temp_y = r_y
scaler_temp = StandardScaler()
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
    temp_x, temp_y, test_size=0.1)
rf_model = RandomForestClassifier(n_estimators=1500, max_features='auto')
rf_model.fit(X_temp_train, y_temp_train)
rf_predicitons = rf_model.predict(X_temp_test)
joblib.dump(rf_model, 'rf_model.pkl', compress=9)
print("done")
