import torch
from sklearn.metrics import mean_squared_error, r2_score
from ann_simple import ANNSimple
from ann_savi import ANNSAVI
from ann_savi_learnable import ANNSAVILearnable
from ann_savi_skip import ANNSAVISkip
from ann_savi_skip_learnable import ANNSAVISkipLearnable
from ann_savi_skip_learnable_fn import ANNSAVISkipLearnableFN
from ann_savi_skip_learnable_bi import ANNSAVISkipLearnableBI


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clazz = None
        if algorithm == "ann_simple":
            clazz = ANNSimple
        elif algorithm == "ann_savi":
            clazz = ANNSAVI
        elif algorithm == "ann_savi_learnable":
            clazz = ANNSAVILearnable
        elif algorithm == "ann_savi_skip":
            clazz = ANNSAVISkip
        elif algorithm == "ann_savi_skip_learnable":
            clazz = ANNSAVISkipLearnable
        elif algorithm == "ann_savi_skip_learnable_fn":
            clazz = ANNSAVISkipLearnableFN
        elif algorithm == "ann_savi_skip_learnable_bi":
            clazz = ANNSAVISkipLearnableBI

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_instance = clazz(device, train_x, train_y, test_x, test_y, validation_x, validation_y)
        model_instance.train_model()
        y_hats = model_instance.test()
        r2 = r2_score(test_y, y_hats)
        rmse = mean_squared_error(test_y, y_hats, squared=False)
        pc = model_instance.pc()
        return max(r2,0), rmse, pc