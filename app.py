# # import pandas as pd
# # import joblib
# # import xgboost as xgb
# # import numpy as np
# # from flask import Flask, request, jsonify
# # from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# # # Charger les encodeurs et le modèle
# # label_encoders = joblib.load('label_encoders.pkl')
# # onehot_encoder = joblib.load('onehot_encoder.pkl')
# # model = joblib.load('xgb_car_price_model.pkl')

# # # Liste des colonnes que le modèle attend
# # features_list = [
# #     'Kilométrage', 'Année', 'Boite de vitesses', 'Carburant', 'Puissance fiscale', 
# #     'Nombre de portes', 'Première main', 'Véhicule dédouané', 'Km_par_an', 
# #     'Puissance_x_Âge', 'Premium_Brand', 'Marque_ABARTH', 'Marque_ALFA-ROMEO', 
# #     'Marque_AUDI', 'Marque_AUTRE', 'Marque_BENTLEY', 'Marque_BMW', 'Marque_BYD', 
# #     'Marque_CADILLAC', 'Marque_CHANA', 'Marque_CHANGAN', 'Marque_CHERY', 'Marque_CHEVROLET', 
# #     'Marque_CITROEN', 'Marque_CUPRA', 'Marque_DACIA', 'Marque_DAEWOO', 'Marque_DAIHATSU', 
# #     'Marque_DFSK', 'Marque_DODGE', 'Marque_DS', 'Marque_FAW', 'Marque_FIAT', 'Marque_FORD', 
# #     'Marque_FOTON', 'Marque_GAZ', 'Marque_GEELY', 'Marque_GREAT-WALL', 'Marque_HAFEI', 
# #     'Marque_HONDA', 'Marque_HUMMER', 'Marque_HYUNDAI', 'Marque_INFINITI', 'Marque_ISUZU', 
# #     'Marque_IVECO', 'Marque_JAGUAR', 'Marque_JEEP', 'Marque_KIA', 'Marque_KTM', 'Marque_LANCIA', 
# #     'Marque_LAND-ROVER', 'Marque_LEXUS', 'Marque_LIFAN', 'Marque_MAHINDRA', 'Marque_MASERATI', 
# #     'Marque_MAZDA', 'Marque_MERCEDES-BENZ', 'Marque_MG', 'Marque_MINI', 'Marque_MITSUBISHI', 
# #     'Marque_NISSAN', 'Marque_OPEL', 'Marque_PEUGEOT', 'Marque_PORSCHE', 'Marque_QUAD', 
# #     'Marque_RENAULT', 'Marque_ROVER', 'Marque_SEAT', 'Marque_SIMCA', 'Marque_SKODA', 'Marque_SMART', 
# #     'Marque_SSANGYONG', 'Marque_SUZUKI', 'Marque_TATA', 'Marque_TESLA', 'Marque_TOYOTA', 
# #     'Marque_VOLKSWAGEN', 'Marque_VOLVO', 'Marque_ZOTYE'
# # ]

# # app = Flask(__name__)

# # def process_and_predict(data):
# #     # Appliquer l'encodage des données
# #     data_processed = encode_data(data, fit_mode=False)
    
# #     # Vérifier que toutes les colonnes attendues sont présentes
# #     missing_cols = set(features_list) - set(data_processed.columns)
# #     if missing_cols:
# #         for col in missing_cols:
# #             data_processed[col] = 0  # Ajouter les colonnes manquantes avec valeur par défaut
    
# #     # Ordonner les colonnes exactement comme le modèle les attend
# #     data_processed = data_processed[features_list]
    
# #     # Conversion en format numpy et prédiction
# #     prediction = model.predict(data_processed.values)
# #     return prediction[0]

# # def encode_data(df, fit_mode=False):
# #     # Colonnes à encoder avec LabelEncoder
# #     label_columns = [
# #         'Boite de vitesses', 'Carburant', 'Première main', 'Véhicule dédouané',
# #         'Dvd / cd / mp3', 'Jantes aluminium', 'Airbags', 'Climatisation auto',
# #         'Abs', 'Système de navigation / gps', 'Vitres éléctriques', 'Anti patinage',
# #         'Esp', 'Intérieur cuir', 'Vitres surteintées', 'Anti démarrage',
# #         'Climatisation multizone', 'Ordinateur de bord', 'Régulateur de vitesse',
# #         'Frein de parking automatique', 'Radar de recul', 'Limiteur de vitesse'
# #     ]
    
# #     # Mode prédiction - utiliser les encodeurs chargés
# #     for col in label_columns:
# #         if col in df.columns:
# #             le = label_encoders[col]
# #             # Gérer les nouvelles valeurs inconnues
# #             df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')
# #             # Si 'Unknown' n'existe pas dans les classes, l'ajouter
# #             if 'Unknown' not in le.classes_:
# #                 le.classes_ = np.append(le.classes_, 'Unknown')
# #             df[col] = le.transform(df[col].astype(str))
    
# #     # Encodage OneHot pour Marque
# #     if 'Marque' in df.columns:
# #         marque_encoded = onehot_encoder.transform(df[['Marque']])
# #         marque_encoded_df = pd.DataFrame(
# #             marque_encoded, 
# #             columns=onehot_encoder.get_feature_names_out(['Marque'])
# #         )
# #         df = pd.concat([df, marque_encoded_df], axis=1)
    
# #     # Supprimer les colonnes inutiles si elles existent
# #     cols_to_drop = ['Nom et Marque', 'Marque']
# #     df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore', inplace=True)
    
# #     # Convertir les booléens en int
# #     bool_cols = df.select_dtypes(include='bool').columns
# #     df[bool_cols] = df[bool_cols].astype(int)
    
# #     return df

# # @app.route('/predict', methods=['POST'])
# # def predict_car_price():
# #     try:
# #         data = request.get_json(force=True)
        
# #         if isinstance(data, dict):
# #             data = [data]
            
# #         df = pd.DataFrame(data)
        
# #         # Vérifier les colonnes obligatoires
# #         required_columns = ['Kilométrage', 'Année', 'Boite de vitesses', 'Carburant', 
# #                           'Puissance fiscale', 'Marque']
# #         missing_required = [col for col in required_columns if col not in df.columns]
# #         if missing_required:
# #             return jsonify({
# #                 'error': f'Colonnes obligatoires manquantes: {missing_required}'
# #             }), 400
        
# #         # Calculer les features dérivées si elles manquent
# #         if 'Km_par_an' not in df.columns and 'Année' in df.columns:
# #             current_year = pd.Timestamp.now().year
# #             df['Km_par_an'] = df['Kilométrage'] / (current_year - df['Année'] + 1)
        
# #         if 'Puissance_x_Âge' not in df.columns and 'Année' in df.columns:
# #             current_year = pd.Timestamp.now().year
# #             df['Puissance_x_Âge'] = df['Puissance fiscale'] / (current_year - df['Année'] + 1)
        
# #         if 'Premium_Brand' not in df.columns and 'Marque' in df.columns:
# #             premium_brands = ['AUDI', 'BMW', 'MERCEDES-BENZ', 'LEXUS', 'PORSCHE', 'JAGUAR', 'VOLVO']
# #             df['Premium_Brand'] = df['Marque'].isin(premium_brands).astype(int)
        
# #         price_prediction = process_and_predict(df)
        
# #         return jsonify({
# #             'predicted_price': float(price_prediction),
# #             'status': 'success'
# #         })
        
# #     except Exception as e:
# #         return jsonify({
# #             'error': str(e),
# #             'status': 'error'
# #         }), 500

# # if __name__ == '__main__':
# #     app.run(debug=True, host='0.0.0.0', port=5000)
# # se7i7
# import pandas as pd
# import joblib
# import xgboost as xgb
# import numpy as np
# from flask import Flask, request, jsonify
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# # Charger les encodeurs et le modèle
# label_encoders = joblib.load('label_encoders.pkl')
# onehot_encoder_marque = joblib.load('onehot_encoder_marque.pkl')
# onehot_encoder_modele = joblib.load('onehot_encoder_modele.pkl')
# model = joblib.load('xgb_car_price_model.pkl')

# # Charger la liste complète des colonnes attendues
# features_list = joblib.load('model_columns.pkl')

# app = Flask(__name__)

# def process_and_predict(data):
#     # Appliquer l'encodage des données
#     data_processed = encode_data(data, fit_mode=False)
    
#     # Vérifier que toutes les colonnes attendues sont présentes
#     missing_cols = set(features_list) - set(data_processed.columns)
#     if missing_cols:
#         for col in missing_cols:
#             data_processed[col] = 0  # Ajouter les colonnes manquantes avec valeur par défaut
    
#     # Ordonner les colonnes exactement comme le modèle les attend
#     data_processed = data_processed[features_list]
    
#     # Conversion en format numpy et prédiction
#     prediction = model.predict(data_processed.values)
#     return prediction[0]

# def encode_data(df, fit_mode=False):
#     # Colonnes à encoder avec LabelEncoder
#     label_columns = [
#         'Boite de vitesses', 'Carburant', 'Première main', 'Véhicule dédouané',
#         'Dvd / cd / mp3', 'Jantes aluminium', 'Airbags', 'Climatisation auto',
#         'Abs', 'Système de navigation / gps', 'Vitres éléctriques', 'Anti patinage',
#         'Esp', 'Intérieur cuir', 'Vitres surteintées', 'Anti démarrage',
#         'Climatisation multizone', 'Ordinateur de bord', 'Régulateur de vitesse',
#         'Frein de parking automatique', 'Radar de recul', 'Limiteur de vitesse'
#     ]
    
#     # Mode prédiction - utiliser les encodeurs chargés
#     for col in label_columns:
#         if col in df.columns:
#             le = label_encoders[col]
#             # Gérer les nouvelles valeurs inconnues
#             df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')
#             # Si 'Unknown' n'existe pas dans les classes, l'ajouter
#             if 'Unknown' not in le.classes_:
#                 le.classes_ = np.append(le.classes_, 'Unknown')
#             df[col] = le.transform(df[col].astype(str))
    
#     # Encodage OneHot pour Marque
#     if 'Marque' in df.columns:
#         marque_encoded = onehot_encoder_marque.transform(df[['Marque']])
#         marque_cols = [f"Marque_{x}" for x in onehot_encoder_marque.categories_[0]]
#         marque_encoded_df = pd.DataFrame(marque_encoded, columns=marque_cols)
#         df = pd.concat([df, marque_encoded_df], axis=1)
    
#     # Encodage OneHot pour Model
#     if 'Model' in df.columns:
#         modele_encoded = onehot_encoder_modele.transform(df[['Model']])
#         modele_cols = [f"Model_{x}" for x in onehot_encoder_modele.categories_[0]]
#         modele_encoded_df = pd.DataFrame(modele_encoded, columns=modele_cols)
#         df = pd.concat([df, modele_encoded_df], axis=1)
    
#     # Supprimer les colonnes inutiles si elles existent
#     cols_to_drop = ['Nom et Marque', 'Marque', 'Model']
#     df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore', inplace=True)
    
#     # Convertir les booléens en int
#     bool_cols = df.select_dtypes(include='bool').columns
#     df[bool_cols] = df[bool_cols].astype(int)
    
#     return df

# @app.route('/predict', methods=['POST'])
# def predict_car_price():
#     try:
#         data = request.get_json(force=True)
        
#         if isinstance(data, dict):
#             data = [data]
            
#         df = pd.DataFrame(data)
        
#         # Vérifier les colonnes obligatoires
#         required_columns = ['Kilométrage', 'Année', 'Boite de vitesses', 'Carburant', 
#                           'Puissance fiscale', 'Marque', 'Model']
#         missing_required = [col for col in required_columns if col not in df.columns]
#         if missing_required:
#             return jsonify({
#                 'error': f'Colonnes obligatoires manquantes: {missing_required}'
#             }), 400
        
#         # Calculer les features dérivées si elles manquent
#         if 'Km_par_an' not in df.columns and 'Année' in df.columns:
#             current_year = pd.Timestamp.now().year
#             df['Km_par_an'] = df['Kilométrage'] / (current_year - df['Année'] + 1)
        
#         if 'Puissance_x_Âge' not in df.columns and 'Année' in df.columns:
#             current_year = pd.Timestamp.now().year
#             df['Puissance_x_Âge'] = df['Puissance fiscale'] / (current_year - df['Année'] + 1)
        
#         if 'Premium_Brand' not in df.columns and 'Marque' in df.columns:
#             premium_brands = ['AUDI', 'BMW', 'MERCEDES-BENZ', 'LEXUS', 'PORSCHE', 'JAGUAR', 'VOLVO']
#             df['Premium_Brand'] = df['Marque'].isin(premium_brands).astype(int)
        
#         price_prediction = process_and_predict(df)
        
#         return jsonify({
#             'predicted_price': float(price_prediction),
#             'status': 'success'
#         })
        
#     except Exception as e:
#         return jsonify({
#             'error': str(e),
#             'status': 'error'
#         }), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)




# # import pandas as pd
# # import joblib
# # import numpy as np
# # from flask import Flask, request, jsonify
# # from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# # # Charger les encodeurs et le modèle
# # label_encoders = joblib.load('label_encoders.pkl')
# # onehot_encoder_marque = joblib.load('onehot_encoder_marque.pkl')
# # onehot_encoder_modele = joblib.load('onehot_encoder_modele.pkl')
# # model = joblib.load('xgb_car_price_model.pkl')

# # # Charger la liste complète des colonnes attendues
# # features_list = joblib.load('model_columns.pkl')

# # app = Flask(__name__)

# # def process_and_predict(data):
# #     # Appliquer l'encodage des données
# #     data_processed = encode_data(data, fit_mode=False)
    
# #     # Vérifier que toutes les colonnes attendues sont présentes
# #     missing_cols = set(features_list) - set(data_processed.columns)
# #     if missing_cols:
# #         for col in missing_cols:
# #             data_processed[col] = 0  # Ajouter les colonnes manquantes avec valeur par défaut
    
# #     # Ordonner les colonnes exactement comme le modèle les attend
# #     data_processed = data_processed[features_list]
    
# #     # Conversion en format numpy et prédiction
# #     log_prediction = model.predict(data_processed.values)
# #     return log_prediction[0]

# # def encode_data(df, fit_mode=False):
# #     # Colonnes à encoder avec LabelEncoder
# #     label_columns = [
# #         'Boite de vitesses', 'Carburant', 'Première main', 'Véhicule dédouané'
# #     ]
    
# #     # Mode prédiction - utiliser les encodeurs chargés
# #     for col in label_columns:
# #         if col in df.columns:
# #             le = label_encoders[col]
# #             # Gérer les nouvelles valeurs inconnues
# #             df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')
# #             if 'Unknown' not in le.classes_:
# #                 le.classes_ = np.append(le.classes_, 'Unknown')
# #             df[col] = le.transform(df[col].astype(str))
    
# #     # Encodage OneHot pour Marque
# #     if 'Marque' in df.columns:
# #         marque_encoded = onehot_encoder_marque.transform(df[['Marque']])
# #         marque_cols = [f"Marque_{x}" for x in onehot_encoder_marque.categories_[0]]
# #         marque_encoded_df = pd.DataFrame(marque_encoded, columns=marque_cols)
# #         df = pd.concat([df, marque_encoded_df], axis=1)
    
# #     # Encodage OneHot pour Model
# #     if 'Model' in df.columns:
# #         modele_encoded = onehot_encoder_modele.transform(df[['Model']])
# #         modele_cols = [f"Model_{x}" for x in onehot_encoder_modele.categories_[0]]
# #         modele_encoded_df = pd.DataFrame(modele_encoded, columns=modele_cols)
# #         df = pd.concat([df, modele_encoded_df], axis=1)
    
# #     # Calculer les features dérivées
# #     current_year = pd.Timestamp.now().year
# #     if 'Année' in df.columns:
# #         df['Km_par_an'] = df['Kilométrage'] / (current_year - df['Année'] + 1)
# #         df['Puissance_x_Âge'] = df['Puissance fiscale'] / (current_year - df['Année'] + 1)
    
# #     if 'Marque' in df.columns:
# #         premium_brands = ['AUDI', 'BMW', 'MERCEDES-BENZ', 'LEXUS', 'PORSCHE', 'JAGUAR', 'VOLVO']
# #         df['Premium_Brand'] = df['Marque'].isin(premium_brands).astype(int)
    
# #     # Supprimer les colonnes inutiles
# #     cols_to_drop = ['Nom et Marque', 'Marque', 'Model']
# #     df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore', inplace=True)
    
# #     return df

# # @app.route('/predict', methods=['POST'])
# # def predict_car_price():
# #     try:
# #         data = request.get_json(force=True)
        
# #         if isinstance(data, dict):
# #             data = [data]
            
# #         df = pd.DataFrame(data)
        
# #         # Vérifier les colonnes obligatoires
# #         required_columns = ['Kilométrage', 'Année', 'Boite de vitesses', 'Carburant', 
# #                           'Puissance fiscale', 'Marque', 'Model']
# #         missing_required = [col for col in required_columns if col not in df.columns]
# #         if missing_required:
# #             return jsonify({
# #                 'error': f'Colonnes obligatoires manquantes: {missing_required}'
# #             }), 400
        
# #         # Obtenir la prédiction logarithmique
# #         log_prediction = process_and_predict(df)
        
# #         # Convertir en prix réel (exponentielle)
# #         predicted_price = np.expm1(log_prediction)  # Equivalent à exp(prediction) - 1
        
# #         return jsonify({
# #             'predicted_price': float(predicted_price),
# #             'status': 'success'
# #         })
        
# #     except Exception as e:
# #         return jsonify({
# #             'error': str(e),
# #             'status': 'error'
# #         }), 500

# # if __name__ == '__main__':
# #     app.run(debug=True, host='0.0.0.0', port=5000)
































from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/predict": {"origins": "https://voiture-prediction.vercel.app/"}})
# 🔁 Chargement du modèle et du préprocesseur
model = joblib.load('xgb_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# 🧠 Les colonnes utilisées dans le prétraitement
categorical_cols = ['marque', 'type de carburant', 'modèle', 'Transmision']
numerical_cols = ['kilométrage', 'année', 'puissance fiscale']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 🔍 Récupérer les données JSON depuis la requête
        data = request.get_json()
          # 📝 Mettre en minuscules les valeurs des colonnes catégorielles
        for col in ['marque', 'type de carburant', 'modèle', 'Transmision']:
            if col in data:
                data[col] = data[col].lower()
        # 🧾 Convertir en DataFrame
        input_df = pd.DataFrame([data])

        # ✅ Vérifier que toutes les colonnes nécessaires sont présentes
        missing_cols = [col for col in categorical_cols + numerical_cols if col not in input_df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {missing_cols}'}), 400

        # 🔄 Appliquer le même prétraitement qu’à l’entraînement
        X_transformed = preprocessor.transform(input_df)

        # 🤖 Prédiction du modèle
        prediction = model.predict(X_transformed)

        # 🧾 Retourner la prédiction
        return jsonify({'predicted_price': round(float(prediction[0]), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
