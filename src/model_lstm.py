import numpy as np

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, RepeatVector

    def train_lstm(X):
        X = np.expand_dims(X, axis=1)

        inputs = Input(shape=(1, X.shape[2]))
        encoded = LSTM(8)(inputs)
        decoded = RepeatVector(1)(encoded)
        decoded = LSTM(X.shape[2], return_sequences=True)(decoded)

        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        autoencoder.fit(X, X, epochs=5, batch_size=32, verbose=0)

        recon = autoencoder.predict(X, verbose=0)
        error = np.mean((X - recon) ** 2, axis=(1, 2))

        return error
except ImportError:
    from sklearn.decomposition import PCA

    def train_lstm(X):
        n_components = min(8, X.shape[0], X.shape[1])
        if n_components < 1:
            return np.zeros(X.shape[0])

        model = PCA(n_components=n_components, random_state=42)
        transformed = model.fit_transform(X)
        recon = model.inverse_transform(transformed)
        error = np.mean((X - recon) ** 2, axis=1)

        return error