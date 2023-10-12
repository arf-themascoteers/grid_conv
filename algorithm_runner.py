import torch
from ann_shared import ANNShared
from ann_shared_small import ANNSharedSmall
from ann_shared_spatial import ANNSharedSpatial
from ann_shared_spatial_small import ANNSharedSpatialSmall
from ann_double_shared import ANNDoubleShared
from ann_double_shared_small import ANNDoubleSharedSmall
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
        elif algorithm == "ann_shared_small":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNSharedSmall(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_shared_spatial":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNSharedSpatial(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_shared_spatial_small":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNSharedSpatialSmall(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_double_shared":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNDoubleShared(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
            model_instance.train_model()
            y_hats = model_instance.test()
        elif algorithm == "ann_double_shared_small":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = ANNDoubleSharedSmall(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
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