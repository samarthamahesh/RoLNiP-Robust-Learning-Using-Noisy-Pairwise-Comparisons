import tensorflow as tf
from sklearn.model_selection import train_test_split
from dataclass import *

def noise_rate_estimation(dataclass):

    X_data, y_data = dataclass.data_x, dataclass.data_y
    X_data = np.hstack((X_data[:, 0, :], X_data[:, 1, :]))

    print(X_data.shape, y_data.shape)

    train_X_data, test_X_data, train_y_data, _ = train_test_split(X_data, y_data, test_size=0.2)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, kernel_initializer='he_normal', activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train_X_data, train_y_data, epochs=15)

    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])

    probs = probability_model.predict(test_X_data)

    eta_thresh_0, eta_thresh_1 = np.percentile(probs[:, 0], 95, interpolation='higher'), np.percentile(probs[:, 1], 95, interpolation='higher')
    robust_eta_0, robust_eta_1 = probs[:, 0], probs[:, 1]
    robust_eta_0[robust_eta_0 >= eta_thresh_0] = 0.0
    robust_eta_1[robust_eta_1 >= eta_thresh_1] = 0.0
    idx_best_0, idx_best_1 = np.argmax(robust_eta_0), np.argmax(robust_eta_1)

    _, noise_SD = probs[idx_best_0, 0], probs[idx_best_0, 1]
    noise_DS, _ = probs[idx_best_1, 0], probs[idx_best_1, 1]
    
    return noise_DS, noise_SD
