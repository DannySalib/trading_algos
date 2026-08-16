import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
    
# This is trash :)

def train_tree(ohlc: pd.DataFrame):

    log_c = np.log(ohlc['close'])

    diff6 = log_c.diff(6)
    diff24 = log_c.diff(24)
    diff168 = log_c.diff(168)

    # -1 or 1 if next 24 hours go up/down
    target = np.sign(log_c.diff(24).shift(-24))

    # Transform to -1, 1 to 0, 1
    target = (target + 1) / 2
    

    dataset = pd.concat([diff6, diff24, diff168, target], axis=1)
    dataset.columns = ['diff6', 'diff24', 'diff168', 'target']
    
    train_data = dataset.dropna()
    train_x = train_data[['diff6', 'diff24', 'diff168']].to_numpy()
    train_y = train_data['target'].astype(int).to_numpy()

    model = DecisionTreeClassifier(min_samples_leaf=5, random_state=69)

    model.fit(train_x, train_y)
    return model

def tree_strategy(ohlc: pd.DataFrame, model):
    log_c = np.log(ohlc['close'])

    diff6 = log_c.diff(6)
    diff24 = log_c.diff(24)
    diff168 = log_c.diff(168)
    
    dataset = pd.concat([diff6, diff24, diff168], axis=1)
    dataset.columns = ['diff6', 'diff24', 'diff168']

    dataset = dataset.dropna()

    insample_pred = model.predict(dataset.to_numpy())
    insample_pred = pd.Series(insample_pred, index=dataset.index)

    # Reindex to actual data
    insample_pred = insample_pred.reindex(ohlc.index)

    # Make predictions tradable
    insample_signal = np.where(insample_pred > 0, 1, -1)
    insample_signal = pd.Series(insample_signal, index=ohlc.index)

    # Get profit factor of signal
    r = log_c.diff().shift(-1)
    rets = insample_signal * r
    insample_pf = rets[rets>0].sum() / rets[rets<0].abs().sum()
    return insample_signal, insample_pf
    

if __name__ == '__main__':
    df = pd.read_parquet('BTCUSD3600.pq')
    df.index = df.index.astype('datetime64[s]')

    df['r'] = np.log(df['close']).diff().shift(-1)

    train_df = df[(df.index.year >= 2016) & (df.index.year < 2020)]

    nn = train_tree(train_df)
    is_sig, is_pf = tree_strategy(train_df, nn)
    print(is_pf)

    (train_df['r'] * is_sig).cumsum().plot()
    plt.show()
    
