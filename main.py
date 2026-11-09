NR_OF_SAMPLES = 3

def real(w,x,b):
    return w*x + b


def predict(w,x,b):
    return w*x + b


def loss(true1,predicted1):
    return (true1 - predicted1) ** 2


def compute_gradient(true, pred, x):
    gradient_w = -2 * x * (true - pred)
    gradient_b = -2 * (true - pred)
    return gradient_w, gradient_b


predictions = []
true_values = []
for i in range(NR_OF_SAMPLES):
    pred = predict(1, 1 + i, 0)
    predictions.append(pred)
    true = real(2, 1 + i, 1)
    true_values.append(true)
    grad_w, grad_b = compute_gradient(true, pred, 1 + i)
    print(grad_w, grad_b)

error = 0
for i in range(NR_OF_SAMPLES):
    error += loss(true_values[i], predictions[i])
mean_squared_error = error / NR_OF_SAMPLES
print("Mean Squared Error:", mean_squared_error)
