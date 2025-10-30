import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(X, y, scaler, nn):

    if X.shape[1] != 2:
        print(f'Cannot print higher dimensional classifiers (2d max).')
        return

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    h = max(x_max-x_min, y_max-y_min) / 1000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    X_mesh_norm = scaler.transform(X_mesh)
    Z = nn.predict(X_mesh_norm)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()