from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="ann_shared", folds=10, algorithms=["ann_shared"])
    c.process()