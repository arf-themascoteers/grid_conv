from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="ann_savi_learnable2", folds=10, algorithms=[
        "ann_savi_learnable"
    ])
    c.process()