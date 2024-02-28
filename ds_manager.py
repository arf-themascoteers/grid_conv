import pandas as pd
from sklearn.model_selection import KFold
import torch
from sklearn import model_selection
from csv_processor import CSVProcessor


class DSManager:
    def __init__(self, csv, folds=10):
        torch.manual_seed(0)
        non_band_columns, band_columns = CSVProcessor.get_ml_columns()
        self.df = pd.read_csv(csv)
        self.x = non_band_columns + band_columns
        self.y = "som"
        self.folds = folds
        columns = self.x + [self.y]
        self.df = self.df[columns]
        self.df = self.df.sample(frac=1)

    def get_k_folds(self):
        kf = KFold(n_splits=self.folds)
        for i, (train_index, test_index) in enumerate(kf.split(self.df)):
            train_data = self.df.iloc[train_index]
            train_data, validation_data = model_selection.train_test_split(train_data, test_size=0.1, random_state=2)
            test_data = self.df.iloc[test_index]
            train_x = train_data[CSVProcessor.get_computable_columns()].to_numpy()
            train_y = train_data["som"].to_numpy()
            test_x = test_data[CSVProcessor.get_computable_columns()].to_numpy()
            test_y = test_data["som"].to_numpy()
            validation_x = validation_data[CSVProcessor.get_computable_columns()].to_numpy()
            validation_y = validation_data["som"].to_numpy()

            yield train_x, train_y, test_x, test_y, validation_x, validation_y

    def get_folds(self):
        return self.folds

