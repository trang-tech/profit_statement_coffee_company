# #%%writefile transform_data.py

# def transform(df):
    
#     df2 = df.copy()
    
#     # Load encoder
#     encoder = 'le.pkl'
#     with open(encoder, 'rb') as f:
#         le = pickle.load(f)
  
#     #Encode the categorical data      
#     df2_encoded = le.transform(df2)
    
    
#     # Load scaler
#     scaler = 'scaler.pkl'
#     with open(scaler, 'rb') as f:
#         scaler = pickle.load(f)
       
#     # Scale the numerical data
#     df2_scaled = scaler.transform(df2_encoded)
    
#     return df2_transformed