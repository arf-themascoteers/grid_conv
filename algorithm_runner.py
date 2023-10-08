import torch
from ann_shared import ANNShared
from ann_centre_only import ANNCentre
from ann_top_left_only import ANNTopLeft
from ann_weighted import ANNWeighted
from ann_weighted2 import ANNWeighted2
from ann_weighted_position import ANNWeightedPosition
from ann_avg import ANNAvg
from ann_avg_skip import ANNAvgSkip
from ann_centric_avg import ANNCentricAvg
from ann_learnable_avg import ANNLearnableAvg
from ann_learnable_avg_skip import ANNLearnableAvgSkip
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


class AlgorithmRunner:
    @staticmethod
    def calculate_score(train_x, train_y,
                        test_x, test_y,
                        validation_x,
                        validation_y,
                        algorithm
                        ):
        y_hats = None
        print(f"Train: {len(train_y)}, Test: {len(test_y)}, Validation: {len(validation_y)}")
        if algorithm == "ann_shared":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNShared(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_centre_only":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNCentre(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_top_left_only":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNTopLeft(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_weighted":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNWeighted(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_weighted2":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNWeighted2(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_weighted_position":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNWeightedPosition(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_avg":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNAvg(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_centric_avg":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNCentricAvg(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_learnable_avg":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNLearnableAvg(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_avg_skip":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNAvgSkip(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_learnable_avg_skip":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNLearnableAvgSkip(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()

        else:
            model_instance = None
            if algorithm == "mlr":
                model_instance = LinearRegression()
            elif algorithm == "plsr":
                size = train_x.shape[1]//2
                if size == 0:
                    size = 1
                model_instance = PLSRegression(n_components=size)
            elif algorithm == "rf":
                model_instance = RandomForestRegressor(max_depth=4, n_estimators=100)
            elif algorithm == "svr":
                model_instance = SVR()

            model_instance = model_instance.fit(train_x, train_y)
            y_hats = model_instance.predict(test_x)

        r2 = r2_score(test_y, y_hats)
        rmse = mean_squared_error(test_y, y_hats, squared=False)
        return max(r2,0), rmse