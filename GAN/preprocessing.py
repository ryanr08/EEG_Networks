import numpy as np

def data_prep(X, y, sub_sample, average, noise):
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:, :, 0:500]
    #print('Shape of X after trimming:', X.shape)


    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    total_X = X_max
    total_y = y
    #print('Shape of X after maxpooling:', total_X.shape)


    # Averaging + noise
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average), axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    #print('Shape of X after averaging+noise and concatenating:', total_X.shape)


    # Subsampling
    for i in range(sub_sample):
        X_subsample = X[:, :, i::sub_sample] + \
                      (np.random.normal(0.0, 0.5, X[:, :, i::sub_sample].shape) if noise else 0.0)

        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))

    #print('Shape of X after subsampling and concatenating:', total_X.shape)
    return total_X/100, total_y


