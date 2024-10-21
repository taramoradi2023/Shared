import keras
import pandas as pd
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
# # from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np


df = pd.read_csv('/home/tara/projects/Directional_coupler/T871/Processed_Data/final_processed_data.csv')

print('Normalized DataFrame:')
print(df)

#Split into features and target
X = df.drop('R', axis = 1)
y = df['R']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#Scale data, otherwise model will fail.
#Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Experiment with deeper and wider networks
model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='linear'))


model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=100, activation='relu'))

model.add(Dense(units=1, activation='linear'))

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
loss = keras.losses.mean_squared_error
metric = keras.metrics.RootMeanSquaredError
model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=70, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs= 200, batch_size = 128, callbacks=[early_stop], validation_data=(X_test_scaled, y_test))
model.summary()

history = model.fit(X_train_scaled, y_train, validation_split=0.25, epochs =50)

from matplotlib import pyplot as plt
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


###########################################
#Predict on test data
predictions = model.predict(X_test_scaled[:5])
print("Predicted values are: ", predictions)
print("Real values are: ", y_test[:5])
#############################################

# Comparison with other models..
# Neural network - from the current code
mse_neural, mae_neural = model.evaluate(X_test_scaled, y_test)
print('Mean squared error from neural net: ', mse_neural)
print('Mean absolute error from neural net: ', mae_neural)

####################################################################
#Linear regression
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

## Linear regression
lr_model = linear_model.LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_score_lr = r2_score(y_test, y_pred_lr)
print('r2 score from linear regression: ', r2_score_lr)
print('Mean squared error from linear regression: ', mse_lr)
print('Mean absolute error from linear regression: ', mae_lr)

###########################################################
## Decision tree
tree = DecisionTreeRegressor()
tree.fit(X_train_scaled, y_train)
y_pred_tree = tree.predict(X_test_scaled)
mse_dt = mean_squared_error(y_test, y_pred_tree)
mae_dt = mean_absolute_error(y_test, y_pred_tree)
r2_score_dt = r2_score(y_test, y_pred_tree)
print('r2 score using decision tree: ', r2_score_dt)
print('Mean squared error using decision tree: ', mse_dt)
print('Mean absolute error using decision tree: ', mae_dt)

#############################################
#Random forest.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
model = RandomForestRegressor(n_estimators = 30, random_state=30)#, max_depth=100)
model.fit(X_train_scaled, y_train)

y_pred_RF = model.predict(X_test_scaled)

mse_RF = mean_squared_error(y_test, y_pred_RF)
mae_RF = mean_absolute_error(y_test, y_pred_RF)
r2_score_RF = r2_score(y_test, y_pred_RF)
print('r2 score using Random Forest:', r2_score_RF)
print('Mean squared error using Random Forest: ', mse_RF)
print('Mean absolute error Using Random Forest: ', mae_RF)

#Feature ranking...
import pandas as pd
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

print("Sample of Predicted Data: ", y_pred_RF[:1])
print("Sample of Real Data: ", y_test[:1])

trace_predicted = go.Scatter(
    x=np.arange(len(y_pred_RF)),
    y=y_pred_RF,
    mode='markers',
    name='Predicted Data',
    marker=dict(color='red', size=6)
)

trace_real = go.Scatter(
    x=np.arange(len(y_test)),
    y=y_test,
    mode='markers',
    name='Real Data',
    marker=dict(color='blue', size=8)  # Adjust size for real data
)
layout = go.Layout(
    title='Real vs Predicted Data (Random Forest)',
    xaxis=dict(title='Data Points'),
    yaxis=dict(title='Values'),
    showlegend=True
)

fig = go.Figure(data=[trace_real, trace_predicted], layout=layout)

pio.show(fig)
