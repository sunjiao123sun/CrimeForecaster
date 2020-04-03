import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn import metrics
import pickle


train_path = '../CrimeForecaster/CRIME-CHICAGO/8/train.npz'
test_path = '../CrimeForecaster/CRIME-CHICAGO/8/test.npz'

train = np.load(train_path, allow_pickle=True)
x_train, y_train = train["x"], train["y"]

test = np.load(test_path, allow_pickle=True)
x_test, y_test = test["x"], test["y"]

shape_x_tra = np.shape(x_train)
x_train = np.reshape(x_train, (shape_x_tra[0] * shape_x_tra[1], -1))
x_test = np.reshape(x_test, (x_test.shape[0] * x_test.shape[1], -1))
print("x_test", np.shape(x_test))
y_test = np.reshape(y_test, (y_test.shape[0], -1))
print("y_test", np.shape(y_test))
# print(np.shape(x_train)[1])

final_pred = []
for i in range(0, np.shape(x_train)[1]):
    #     print("i: ", i)
    train_inner, test_inner = x_train[:, i], y_test[:, i]
    history = [x for x in train_inner]
    predictions = list()
    if min(history)!= 1 and max(history) != 0:
        #         print("normal")
        for t in range(len(test_inner)):
            model = ARIMA(history, order=(1,0,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0][0]
            predictions.append(yhat)
            obs = test_inner[t]
            history.append(obs)
    if max(history) == 0:
        #         print("all zero")
        predictions += [0 for nu in range(len(test_inner))]
    if min(history) == 1:
        #         print("all one")
        predictions += [1 for nu in range(0, len(test_inner))]
    final_pred.append(predictions)

final_pred = np.array(final_pred)
print(np.shape(final_pred))
final_pred = final_pred.transpose()
print(np.shape(final_pred))

final_pred[final_pred >= 0.5] = 1
final_pred[final_pred <= 0.5] = 0
# print(final_pred)
print("macro-f1:", metrics.f1_score(y_test, final_pred, average = 'macro'))
print("micro-f1:", metrics.f1_score(y_test, final_pred, average = 'micro'))

np.savez_compressed(
    'result/chicago_12.npz',
    predict = final_pred,
    ground_truth = y_test,
    micro = metrics.f1_score(y_test, final_pred, average = 'micro'),
    marco = metrics.f1_score(y_test, final_pred, average = 'macro')
)



