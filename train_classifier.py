import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Function to pad or truncate the feature vector to the expected length
def pad_or_truncate(feature_vector, expected_length):
    if len(feature_vector) < expected_length:
        # Pad feature vector with zeros
        padded_vector = feature_vector + [0] * (expected_length - len(feature_vector))
        return padded_vector
    elif len(feature_vector) > expected_length:
        # Truncate feature vector
        truncated_vector = feature_vector[:expected_length]
        return truncated_vector
    else:
        return feature_vector

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = data_dict['labels']

# Determine the expected length of feature vectors
expected_length = len(data[0])

# Pad or truncate all feature vectors to ensure they have the same length
data_padded = [pad_or_truncate(feature_vector, expected_length) for feature_vector in data]

# Convert the list of feature vectors to a NumPy array
data_np = np.array(data_padded)

x_train, x_test, y_train, y_test = train_test_split(data_np, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
