import numpy as np
import pickle
from sklearn.model_selection import train_test_split


def Power_Normalization(x):
    for i in range(x.shape[0]):
        max_power = (np.power(x[i, 0, :], 2) + np.power(x[i, 1, :], 2)).max()
        x[i] = x[i] / np.power(max_power, 1 / 2)
    return x


def get_dataset(rx):
    file_path = 'D:/wanh/code/dataset/ManySig.pkl'
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    rx_index = data['rx_list'].index(rx)

    num_transmitters = 6
    num_samples_per_transmitter = 500
    x_dataset = []
    y_dataset = []

    for tx in range(num_transmitters):
        tx_data = data['data'][tx][rx_index][0][0]
        tx_data_swapped = np.transpose(tx_data, (0, 2, 1))
        tx_data_swapped = tx_data_swapped[:num_samples_per_transmitter]
        x_dataset.append(tx_data_swapped)
        y_dataset.extend([tx] * num_samples_per_transmitter)

    x_dataset = np.vstack(x_dataset)
    y_dataset = np.array(y_dataset)

    return x_dataset, y_dataset


def WiFi_ReadDataset(rx, random_num):
    x, y = get_dataset(rx)
    x = Power_Normalization(x)
    y = y.astype(np.uint8)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=random_num)
    return X_train, X_test, Y_train, Y_test


def Domain_Dataset(rx, random_num):
    X_, X_test, Y_, Y_test = WiFi_ReadDataset(rx, random_num)
    X_train, X_val, Y_train, Y_val = train_test_split(X_, Y_, test_size=0.3, random_state=random_num)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


