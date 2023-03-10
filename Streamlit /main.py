import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split 
import seaborn as sns
import matplotlib.pyplot as plt

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()



st.markdown(
	"""
	<style>
	.main {
	background-color: #F5F5F5;
	}
	</style>
	""",
	unsafe_allow_html=True
)



with header:
	st.title('Welcome to my awesome data Analytics project!')
	st.text('In this project I look into the business of a coffee company')

with dataset:
	st.header('Coffe Dataset')
	st.text('I found this dataset on Data Worlds')

	coffee_subset = pd.read_csv('../Data/Clean_data/coffee_subset.csv')
	st.write(coffee_subset.head())


	st.subheader('The Actual Profit and Tagert Profit for Fourth Quarter of 2012 and 2013')
	coffee_profits = pd.DataFrame(coffee_subset['profit'].value_counts()).head(50)
	st.bar_chart(coffee_profits)


	# st.subheader('The Profit Tagert for or Fourth Quarter of 2012 and 2013')
	coffee_target_profit = pd.DataFrame(coffee_subset['target_profit'].value_counts().head(50))
	st.bar_chart(coffee_target_profit)

	st.subheader('The Actual Sales and Target Sales for Fourth Quarter of 2012 and 2013')
	coffee_profits = pd.DataFrame(coffee_subset['sales'].value_counts()).head(50)
	st.line_chart(coffee_profits)


	# st.subheader('The Profit Sales for or Fourth Quarter of 2012 and 2013')
	coffee_target_profit = pd.DataFrame(coffee_subset['target_sales'].value_counts().head(50))
	st.line_chart(coffee_target_profit)










# with features:
# 	st.header('The features I created')
# 	st.markdown('* ** first feature:')






with modelTraining:
	st.header('Time to train the model!')
	st.text('Here you get to choose the hyperparameters of the model and see how the performance changes')


	# sel_col, disp_col = st.columns(2)

	# max_depth = sel_col.slider('What should be the max_depth of the models', min_value =0, max_value=100, step=10)

	# n_estimators = sel_col.selectbox('How much is the minimum profit for each product?', options = [5,10,15,20,25,30,35,40,'Depent on product'], index =0)

	max_depth_info = """
	**Max depth:**

	This parameter controls the maximum depth of each decision tree in the random forest. 
	A higher value can lead to overfitting, while a lower value can lead to underfitting. 
	For small to medium-sized datasets, a value of 5 to 15 is typically a good starting point. 
	For larger datasets, you might need to increase the value to 20 or higher, depending on the complexity of the data.
	"""

	n_estimators_info = """
	**Number of estimators:**

	This parameter controls the number of trees in the random forest. 
	A higher value can lead to better performance, but also increases the computational cost. 
	For small to medium-sized datasets, a value of 50 to 100 is typically a good starting point. 
	For larger datasets, you might need to increase the value to 200 or higher, depending on the complexity of the data.
	"""

	sel_col, disp_col = st.columns(2)

	# Provide guidance on selecting max_depth
	sel_col.markdown(max_depth_info, unsafe_allow_html=True)
	max_depth = sel_col.slider('Select max depth:', min_value=1, max_value=100, step=1, value=5)

	# Provide guidance on selecting n_estimators
	sel_col.markdown(n_estimators_info, unsafe_allow_html=True)

	n_estimators_options = [5, 10, 15, 20, 25, 30, 35, 40,50,60,70,80,90, 'Depent on product']
	n_estimators_selection = sel_col.selectbox('Select minimum profitability threshold:', options=n_estimators_options, index=0)

	if n_estimators_selection == 'Depent on product':
	    model = RandomForestRegressor(max_depth=max_depth)
	else:
	    n_estimators = int(n_estimators_selection)
	    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)




	# sel_col.text('Here is a list of features in my data:')
	# sel_col.write(coffee_subset.columns)

	input_feature = sel_col.text_input('The input features:','target_sales')


	coffee= pd.read_csv('../Data/Clean_data/coffee.csv')
	

	if n_estimators == 'Depent on product':
		model = RandomForestRegressor(max_depth=max_depth)
	else:
		model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)




	X = coffee.drop('target_profit', axis=1)
	y = coffee['target_profit']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




	
	model.fit(X_train, y_train)

	
	y_pred = model.predict(X_test)


			# Generate scatterplot
	sns.scatterplot(y=y_pred, x=y_test)

	# Show the plot using st.pyplot()
	st.pyplot()

	st.set_option('deprecation.showPyplotGlobalUse', False)

	disp_col.subheader('Random Forest Regression Perfomance Metrics:')

	disp_col.subheader('R2:')
	disp_col.write(r2_score(y_test, y_pred))

	disp_col.subheader('MAE:')
	disp_col.write(mean_absolute_error(y_test, y_pred))

	
	disp_col.subheader('RMSE')
	disp_col.write(mean_squared_error(y_test, y_pred, squared=False))

	disp_col.subheader('MSE')
	disp_col.write(mean_squared_error(y_test, y_pred, squared=True))

	st.set_option('deprecation.showPyplotGlobalUse', False)

























