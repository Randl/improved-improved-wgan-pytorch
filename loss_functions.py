import torch
import torch.autograd as autograd
from torch.autograd import Variable


def gradient_penalty(fake_data, real_data, discriminator):
    alpha = torch.cuda.FloatTensor(fake_data.shape[0], 1, 1, 1).uniform_(0, 1).expand(fake_data.shape)
    interpolates = alpha * fake_data + (1 - alpha) * real_data
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates, _ = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def consistency_term(real_data, discriminator, Mtag=0):
    discriminator.drop = True

    d1, d_1 = discriminator(real_data)
    d2, d_2 = discriminator(real_data)

    discriminator.drop = False
    # why max is needed when norm is positive?
    consistency_term = (d1 - d2).norm(2, dim=1) + 0.1 * (d_1 - d_2).norm(2, dim=1) - Mtag
    return consistency_term.mean()
