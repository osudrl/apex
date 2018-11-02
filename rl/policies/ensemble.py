class EnsemblePolicy:
    def __init__(self, vpolicy):
        self.vpolicy = vpolicy # policy vector aka a list of torch models

    # take average action over ensemble of actions
    def act(self, x, deterministic=False):
        return None, sum([policy.act(x, deterministic)[1] for policy in self.vpolicy]) / len(self.vpolicy)