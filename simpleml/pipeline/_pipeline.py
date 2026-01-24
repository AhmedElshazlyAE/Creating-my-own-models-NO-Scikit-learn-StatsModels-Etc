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

def make_pipeline(*objs):
    steps = [(obj.__class__.__name__.lower(), obj) for obj in objs]
    return Pipeline(steps)
