from distributions.base import Distribution

class DiagonalGaussian(Distribution):
    def log_likelihood(self, x, distribution):
        means = distribution["means"]
        log_stds = distribution["log_stds"]
        stds = distribution["stds"]

        var = stds.pow(2)

        log_density = -(x - means).pow(2) / (
            2 * var) - 0.5 * math.log(2 * math.pi) - log_stds

        return log_density.sum(1)

    def kl_divergence(self, mean0, log_std0, std0,
                            mean1, log_std1, std1):
        mean0 = distribution0["means"]
        log_std0 = distribution0["log_stds"]
        std0 = distribution0["stds"]

        mean1 = distribution1["means"]
        log_std1 = distribution1["log_stds"]
        std1 = distribution1["stds"]

        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1)

    def entropy(self, distribution):
        log_std = distribution["log_stds"]

        return (log_stds + np.log(np.sqrt(2 * np.pi * np.e))).sum(1)
