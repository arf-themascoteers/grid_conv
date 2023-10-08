from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="ann_learnable_avg_skip2", folds=10, algorithms=[
        "ann_learnable_avg_skip"
    ])
    c.process()