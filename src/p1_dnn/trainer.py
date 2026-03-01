import numpy as np

from .models import NeuralNetwork


class Trainer:
    def __init__(
        self, model: NeuralNetwork, learning_rate: float = 0.01, batch_size: int = 100
    ):
        self.model = model
        self.lr = learning_rate
        self.batch_size = batch_size

    def train(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        epochs: int,
        val_data: tuple = None,
    ):
        n_samples = train_x.shape[0]
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            self.model.rng.shuffle(indices)
            x_shuffled = train_x[indices]
            y_shuffled = train_y[:, indices]

            for i in range(0, n_samples, self.batch_size):
                x_batch = x_shuffled[i : i + self.batch_size, :].T
                y_batch = y_shuffled[:, i : i + self.batch_size]

                self.model.forward(x_batch)
                self.model.backward(y_batch, self.lr)

            if (epoch + 1) % 10 == 0:
                sr = self.model.success_rate(*val_data)
                print(f"[PROGRESS] Epoch {epoch + 1}/{epochs} - Success Rate: {sr:.4f}")
