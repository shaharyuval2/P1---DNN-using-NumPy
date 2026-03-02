# MNIST from Scratch: Pure NumPy Implementation
## Developed by Shahar — **The Stupid Guy**

This project is a modular, high-performance Deep Neural Network (DNN) built entirely from the ground up using only **NumPy**. It serves as a deep dive into the underlying mechanics of Artificial Intelligence and marks the first major milestone in my journey toward AI research.


### Project Overview

To truly understand AI, I decided to "break the black box." Instead of relying on high-level frameworks like PyTorch or TensorFlow, **I manually derived the backpropagation formulas from scratch** and implemented the vectorized matrix operations myself. This ensured a first-principles understanding of exactly how data/gradients flow through a network.

* **Custom DNN Engine:** Modular architecture allowing for dynamic layer sizes and activation types.
* **Optimized Math:** Fully vectorized batch processing for efficient training on the MNIST dataset.
* **Professional Architecture:** Decoupled "Source Layout" design, treating the engine as a reusable Python package.
* **Interactive Showcase:** A Pygame-based GUI where you can draw digits and watch the model predict them in real-time.

![showcase_demo](https://github.com/user-attachments/assets/720b8978-a082-4bc0-a7eb-ef1b69200c4a)

## The Mathematics

The engine implements **Stochastic Gradient Descent (SGD)**, utilizing specific initialization strategies to ensure stable training dynamics across deep architectures.

### Weight Initialization
To mitigate the **Vanishing** or **Exploding Gradient** problems, the model scales initial weights based on the specific activation function used in the layer:

* **Sigmoid/Tanh (Xavier Initialization):** Optimized for symmetric activations.
    
$$\text{scale} = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}$$

* **ReLU (He Initialization):** Optimized for non-symmetric activations.
    
$$\text{scale} = \sqrt{\frac{2}{n_{\text{in}}}}$$

### Derivation & Mathematical Intuition
The objective of proper initialization is to maintain a constant variance of activations during the forward pass and gradients during the backward pass.

#### 1. Core Assumptions
To simplify the variance analysis, we assume the following:
1. Weights ($$w$$) and inputs ($$x$$) are **Independent and Identically Distributed (I.I.D.)**.
2. Both weights and inputs have a **mean of zero**: $$E[w] = 0$$ and $$E[x] = 0$$.
3. The activation function is operating in a **linear region** (e.g., Sigmoid/Tanh near the origin).

#### 2. Forward Pass Variance
From our first assumption $$y = \sum_{i=1}^{n_{\text{in}}} w_i x_i$$, so using the second and third assumption the variance of the output is:

$$\text{Var}(y) = n_{\text{in}} \cdot \text{Var}(w) \cdot \text{Var}(x)$$

From this, we can derive the scaling factor between layers:
$$\frac{\text{Var}(y)}{\text{Var}(x)} = n_{\text{in}} \cdot \text{Var}(w)$$



#### 3. Addressing Gradient Instability
The ratio of variances determines the numerical stability of the network:
* **If $$\frac{\text{Var}(y)}{\text{Var}(x)} > 1$$:** Variance grows exponentially with depth, leading to **Gradient Explosion**.
* **If $$\frac{\text{Var}(y)}{\text{Var}(x)} < 1$$:** Variance shrinks toward zero, leading to **Gradient Vanishing**.

To keep the variance stable ($$\text{Var}(y) \approx \text{Var}(x)$$), we require:

$$\text{Var}(w) = \frac{1}{n_{\text{in}}}$$

#### 4. Initialization Methods

**Xavier (Glorot) Initialization**
By applying the same logic to the backward pass (backpropagation), we find a second requirement: $$\text{Var}(w) = 1/n_{\text{out}}$$. To satisfy both forward and backward constraints as closely as possible, Xavier initialization uses the harmonic mean:
$$\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \implies \text{scale} = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}$$

**He (Kaiming) Initialization**
The ReLU activation function sets all negative values to zero, which effectively **halves the variance** of the output. To compensate for this loss of signal, the variance requirement is doubled:
$$n_{\text{in}} \cdot \text{Var}(w) = 2 \implies \text{Var}(w) = \frac{2}{n_{\text{in}}}$$

$$\text{scale}_{\text{He}} = \sqrt{\frac{2}{n_{\text{in}}}}$$

and here we neglect the backward constraint

### Backpropagation: The Engine of Learning
here attached my personal notes and derivation of the core formulas of the backpropogation algorithm

[dnn_backprop_derivation.pdf](https://github.com/user-attachments/files/25677832/dnn_backprop_derivation.pdf)

#### The Algorithm

By analyzing the mathematical derivations in the attached notes, we can observe the emergence of a highly efficient recursive structure for the backpropagation algorithm.

First, we define the **Error Term** for any given layer $$l$$ as:

$$\delta^{(l)} = \frac{\partial C_n}{\partial z^{(l)}} = \sigma'(z^{(l)})\cdot\frac{\partial C_n}{\partial a^{(l)}}$$

This term represents how much the weighted input $$z$$ in layer $$l$$ contributes to the total cost. Using this definition, the formulas for the entire network simplify into an elegant recursive process:

**1. Output Layer Error:**
For the final layer $$L$$, using **Softmax** activation function in the final layer paired with a **Cross-Entropy** loss function, the error simplifies to the difference between our prediction and the ground truth:

$$\delta^{(L)} = a^{(L)} - y_{target}$$

**2. The Recursive Step (Error Propagation):**
To find the error in the previous hidden layer, we "push" the error back through the weights of the current layer:

$$\delta^{(l-1)} = ((w^{(l)})^T \cdot \delta^{(l)}) \odot \sigma'(z^{(l-1)})$$

(Where $$\odot$$ represents the element-wise Hadamard product)



**3. Gradient Calculation:**
Once the error terms are known, the partial derivatives for the weights ($$w$$) and biases ($$b$$) are calculated as:

$$\frac{\partial C_n}{\partial w^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^T$$

$$\frac{\partial C_n}{\partial b^{(l)}} = \delta^{(l)}$$

### Installation & Setup

Ensure you have a virtual environment active before proceeding:

1.  **Clone the Repository**
    
```bash
    git clone https://github.com/shaharyuval2/P1---DNN-using-NumPy.git
    cd P1_DNN
```

2. **Create and Activate Virtual Environment**
```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3.  **Install in Editable Mode**
    
```bash
    pip install -e .
```

This uses the `pyproject.toml` configuration to set up the `p1_dnn` package and its dependencies (NumPy, Pygame, Scipy).


### How to Use

1.  **Generate the Dataset**
    Before training, you must download and convert the MNIST data:
    
```bash
    python3 data/generate_mnist_csv.py
```

2.  **Train the Model**
    By running the training script:
    
```bash
    python3 apps/training.py
```

3.  **Live Showcase**
    Launch the interactive drawing board to test your trained model:
    
```bash
    python3 apps/ShowCase.py
```

