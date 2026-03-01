from p1_dnn.models import NeuralNetwork
from p1_dnn.trainer import Trainer
from p1_dnn.utils import one_hot, splitData


def main():
    # Load Data
    print("[INFO] Loading MNIST dataset...")
    train_labels, train_images = splitData("data/mnist_train.csv")
    test_labels, test_images = splitData("data/mnist_test.csv")

    # Preprocess lables
    train_y_one_hot = one_hot(train_labels, 10)

    # Initialize Model
    input_dim = train_images.shape[1]
    model = NeuralNetwork(sizes=[input_dim, 16, 16, 10], activation_type="relu")

    # Check baseline model performance
    initial_sr = model.success_rate(test_labels, test_images)
    print(f"[INFO] Initial Success Rate: {initial_sr:.4f}")

    # Setup Trainer and Run Training
    trainer = Trainer(model=model, learning_rate=0.01, batch_size=100)
    print("[INFO] Starting training loop...")
    trainer.train(
        train_x=train_images,
        train_y=train_y_one_hot,
        epochs=100,
        val_data=(test_labels, test_images),
    )

    # Final Evaluation & Saving
    final_sr = model.success_rate(test_labels, test_images)
    print(f"[SUCCESS] Final Success Rate: {final_sr:.4f}")

    model_name = "models/mnist_dnn_v1.npz"
    model.save(model_name)
    print(f"[INFO] Model weights saved to {model_name}")


if __name__ == "__main__":
    main()
