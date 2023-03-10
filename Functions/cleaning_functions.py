cleaning_functions.py 

def drop_columns(df):
    
    df2 = df.copy()
    df2 = df2[[ 'Date','Product', 'Product Line','Product Type', 'Type', 'Market Size',
                      'Market','State',  'Marketing', 'Total Expenses',                      
                     'Inventory', 'Cogs', 'Margin', 'Sales','Profit',
                      'Difference Between Actual and Target Profit',
                         'Target COGS', 'Target Margin',
                       'Target Sales', 'Target Profit' ]]
    
    return df2


def rename_columns(df):
    
    df2 = df.copy()
    df2.columns = [col.lower().replace(' ', '_') for col in df2.columns]
    
    return df2


def clean_date_column(df):
    
    df2 = df.copy()
    
    df2['date'] = df2['date'].apply(lambda x: x[:7])
    df2['date'] = pd.to_datetime(df2['date']).dt.strftime('%b-%Y')
    
    return df2
  
    
def preprocess_dataframe(df):
    
    df2 = df.copy()
    
    df2 = drop_columns(df2)
    df2 = rename_columns(df2)
    df2 = clean_date_column(df2)
    
    return df2