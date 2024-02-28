from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="ann_savi_skip_learnable", folds=10, algorithms=[
        "ann_savi_skip_learnable"
    ])
    c.process()