import torch


class VaeDistribution:

    def rsample_with_parts(self, shape):
        z = self.rsample(shape)
        return z, None

    def log_prob_from_parts(self, z, data):
        log_prob = self.log_prob(z)
        assert torch.isfinite(log_prob).all()
        return log_prob

    def rsample_log_prob(self, shape):
        z, data = self.rsample_with_parts(shape)
        return z, self.log_prob_from_parts(z, data)


class EuclideanNormal(torch.distributions.Normal, VaeDistribution):

    def log_prob(self, value):
        return super().log_prob(value).sum(dim=-1)

class MultivariateEuclideanNormal(torch.distributions.MultivariateNormal, VaeDistribution):

    def log_prob(self, value):
        return super().log_prob(value).sum(dim=-1)
