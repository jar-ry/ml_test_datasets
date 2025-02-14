import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from helper.data_helper import get_data_path, get_data_path

def train(input_data: list[str], output_data: list[str], is_local) -> pd.DataFrame:
    input_data_assets = get_data_path(input_data, is_local)
    output_data_assets = get_data_path(output_data, is_local=is_local)
    
    x_train = pd.read_csv(input_data_assets['x_train'])
    y_train = pd.read_csv(input_data_assets['y_train'])

    reg = LinearRegression()
    reg.fit(x_train,y_train)


    with open(output_data_assets['lr_model']) as file:
        pickle.dump(reg, file)
