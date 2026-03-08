# simpleml/Pipeline/_pipeline.py
# A simple implementation of a machine learning pipeline
class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        return dict(self.steps)

    def fit(self, X, y=None):
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt)
            else:
                step.fit(Xt)
                Xt = step.transform(Xt)

        # last is estimator
        name, est = self.steps[-1]
        est.fit(Xt, y)
        return self

    def predict(self, X):
        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.transform(Xt)

        name, est = self.steps[-1]
        return est.predict(Xt)
    
    def predict_proba(self, X):
        if not hasattr(self.steps[-1][1], "predict_proba"):
            raise AttributeError(f"The last step '{self.steps[-1][0]}' does not have a 'predict_proba' method.")
        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.transform(Xt)

        name, est = self.steps[-1]
        return est.predict_proba(Xt)

def make_pipeline(*objs):
    steps = [(obj.__class__.__name__.lower(), obj) for obj in objs]
    return Pipeline(steps)
