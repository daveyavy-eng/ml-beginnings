NR_OF_SAMPLES = 3


print("/")
print("//")
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


gradient_w = 0
gradient_b = 0
predictions = []
true_values = []

for i in range(NR_OF_SAMPLES):

    pred = predict(1, 1 + i, 0)
    predictions.append(pred)
    true = real(2, 1 + i, 1)
    true_values.append(true)

    grad_w, grad_b = compute_gradient(true, pred, 1 + i)
    gradient_w += grad_w 
    gradient_b += grad_b

avg_gradient_w = gradient_w / NR_OF_SAMPLES
avg_gradient_b = gradient_b / NR_OF_SAMPLES

error = 0
for i in range(NR_OF_SAMPLES):

    error += loss(true_values[i], predictions[i])
mean_squared_error = error / NR_OF_SAMPLES
print("Mean Squared Error:", mean_squared_error)
print("Average Gradient w:", avg_gradient_w)
print("Average Gradient b:", avg_gradient_b)
