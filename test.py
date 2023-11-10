from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="shared", folds=10, algorithms=[
        "ann_shared_small"
    ])
    c.process()