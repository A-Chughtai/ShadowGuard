import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data1 = pd.read_csv('data1.csv')

received = data1['packets_received']
forwarded = data1['packets_forwarded']
login_attempts = data1['unusual_login_attempts']
shared_resources = data1['resources_shared']
cpu = data1['cpu_usage']
memory = data1['memory_usage']
storage = data1['storage_usage']
uptime = data1['device_uptime']
latency = data1['latency']
bandwidth = data1['bandwidth_usage']
transferred = data1['data_transferred']
label1 = data1['label']

data2 = pd.read_csv('data2.csv')

received2 = data2['packets_received']
forwarded2 = data2['packets_forwarded']
login_attempts2 = data2['unusual_login_attempts']
shared_resources2 = data2['resources_shared']
cpu2 = data2['cpu_usage']
memory2 = data2['memory_usage']
storage2 = data2['storage_usage']
uptime2 = data2['device_uptime']
latency2 = data2['latency']
bandwidth2 = data2['bandwidth_usage']
transferred2 = data2['data_transferred']
label2 = data2['label']

data3 = pd.read_csv('data3.csv')

received3 = data3['packets_received']
forwarded3 = data3['packets_forwarded']
login_attempts3 = data3['unusual_login_attempts']
shared_resources3 = data3['resources_shared']
cpu3 = data3['cpu_usage']
memory3 = data3['memory_usage']
storage3 = data3['storage_usage']
uptime3 = data3['device_uptime']
latency3 = data3['latency']
bandwidth3 = data3['bandwidth_usage']
transferred3 = data3['data_transferred']
label3 = data3['label']


x1 = np.concatenate([received, received2, received3])
x2 = np.concatenate([forwarded, forwarded2, forwarded3])
x3 = np.concatenate([login_attempts, login_attempts2, login_attempts3])
x4 = np.concatenate([shared_resources, shared_resources2, shared_resources3])
x5 = np.concatenate([cpu, cpu2, cpu3])
x6 = np.concatenate([memory, memory2, memory3])
x7 = np.concatenate([storage, storage2, storage3])
x8 = np.concatenate([uptime, uptime2, uptime3])
x9 = np.concatenate([latency, latency2, latency3])
x10 = np.concatenate([bandwidth, bandwidth2, bandwidth3])
x11 = np.concatenate([transferred, transferred2, transferred3])
y = np.concatenate([label1, label2, label3])


x = [x8, x9, x10, x1, x2, x11, x4, x5, x6, x7, x3]
x = np.hstack([arr.reshape(-1, 1) for arr in x])

print(x.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(11,)),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


model.summary()


# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.2, 
    class_weight=class_weights_dict, 
    callbacks=[early_stopping],
    verbose=1
)


# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))


model.save('trained_model.keras')