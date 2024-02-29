from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(prefix="mixed", folds=3, algorithms=[
        "ann_savi",
        "ann_savi_skip",

        "ann_savi_learnable",
        "ann_savi_skip_learnable",
        "ann_savi_skip_learnable_fn",
        "ann_savi_skip_learnable_bi",
    ])
    c.process()