import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data = pd.read_csv('KL_property.csv')

# Clean the column names (remove spaces)
data.columns = data.columns.str.strip()

# Prepare the input features and target variable
X = data[['Location', 'Rooms', 'Size']]  # Adjusted features based on user input
y = data['Price']  # Target variable

# Label encode the 'Location' column
label_encoder = LabelEncoder()
X['Location'] = label_encoder.fit_transform(X['Location'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the ANN model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=3))  # Input layer with 3 features
model.add(Dense(units=32, activation='relu'))  # Hidden layer
model.add(Dense(units=1, activation='linear'))  # Output layer with linear activation for regression
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)

# Print the Mean Absolute Error (MAE)
print(f"Mean Absolute Error on Test Set: {mae:.2f}")


# Example user inputs for prediction: location, rooms, size
user_inputs = [['Taman Tun Dr Ismail, Kuala Lumpur', 4, 5000]]  # Example user input

# Convert user inputs to numerical values (similar to training data preprocessing)
user_inputs_df = pd.DataFrame(user_inputs, columns=['Location', 'Rooms', 'Size'])
user_inputs_df['Location'] = label_encoder.transform(user_inputs_df['Location'])

# Make predictions
predicted_prices = model.predict(user_inputs_df)

# Display predicted prices
print("Predicted House Prices:")
for pred in predicted_prices:
    print(f"RM{pred[0]:,.2f}")
