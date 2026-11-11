NR_OF_SAMPLES = 3
LEARNING_RATE = 0.0001

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


def MSE(true_values, predictions):
    error = 0
    for i in range(len(true_values)):
        error += loss(true_values[i], predictions[i])
    return error / len(true_values)


prediction_weight = 1
prediction_bias = 0

gradient_w = 0
gradient_b = 0

iteration = 0
MAX_ITERATIONS = 1000000

mean_squared_error = 1
while mean_squared_error > 0.0001 and iteration < MAX_ITERATIONS:
    iteration += 1

    predictions = []
    true_values = []

    gradient_w = 0
    gradient_b = 0

    for i in range(NR_OF_SAMPLES):

        pred = predict(prediction_weight, 1 + i, prediction_bias)
        predictions.append(pred)
        true = real(2, 1 + i, 1)
        true_values.append(true)

        grad_w, grad_b = compute_gradient(true, pred, 1 + i)
        gradient_w += grad_w 
        gradient_b += grad_b

    avg_gradient_w = gradient_w / NR_OF_SAMPLES
    avg_gradient_b = gradient_b / NR_OF_SAMPLES


    mean_squared_error = MSE(true_values, predictions)

    prediction_weight -= LEARNING_RATE * avg_gradient_w
    prediction_bias -= LEARNING_RATE * avg_gradient_b

    predictions.clear()
    for i in range(NR_OF_SAMPLES):
        pred = predict(prediction_weight, 1 + i, prediction_bias)
        predictions.append(pred)

    mean_squared_error = MSE(true_values, predictions)

print("Iterations:", iteration)
print("Weight:", prediction_weight)
print("Bias:", prediction_bias)
print("Mean Squared Error:", mean_squared_error)
