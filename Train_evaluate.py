import numpy as np
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

def train(model, X_train, X_test, y_train, y_test, n_epochs = 2000, batch_size = 100):
    optimizer = optim.Adam(model.parameters())
    loss_f = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=batch_size)
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_f(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 50 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_f(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_f(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    return model


def evaluate(model, X_train, X_test, y_train, y_test):
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_f(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_f(y_pred, y_test))
    print("Train RMSE:", train_rmse.item())
    print("Test RMSE:", test_rmse.item())
    
    
def visualize_result(dataset, X_train, X_test, model):
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(dataset) * np.nan
        y_pred = model(X_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(X_train)[:, -1, :]
        # shift test predictions for plotting
        test_plot = np.ones_like(dataset) * np.nan
        test_plot[train_size+lookback:len(dataset)] = model(X_test)[:, -1, :]
    # plot
    plt.plot(dataset, c='b')
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()    
        

    