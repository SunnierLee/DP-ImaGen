import numpy as np
import torch as pt
from collections import namedtuple

rff_param_tuple = namedtuple('rff_params', ['w', 'b'])


def flat_data(data, labels, device, n_labels=10, add_label=False):
    bs = data.shape[0]
    if add_label:
        gen_one_hots = pt.zeros(bs, n_labels, device=device)
        gen_one_hots.scatter_(1, labels[:, None], 1)
        labels = gen_one_hots
        return pt.cat([pt.reshape(data, (bs, -1)), labels], dim=1)
    else:
        if len(data.shape) > 2:
            return pt.reshape(data, (bs, -1))
        else:
            return data


def calculate_norm(x_r, x_i):
    return pt.sqrt(pt.mul(x_r, x_r) + pt.mul(x_i, x_i))


def rff_sphere(x, rff_params):
    """
    this is a Pytorch version of anon's code for RFFKGauss
    Fourier transform formula from http://mathworld.wolfram.com/FourierTransformGaussian.html
    """
    w = rff_params.w
    xwt = pt.mm(x, w.t())
    z_1 = pt.cos(xwt)
    z_2 = pt.sin(xwt)
    z_cat = pt.cat((z_1, z_2), 1)
    norm_const = pt.sqrt(pt.tensor(w.shape[0]).to(pt.float32))
    z = z_cat / norm_const  # w.shape[0] == n_features / 2
    return z


def weights_sphere(d_rff, d_enc, sig, device):
    w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(sig)).to(pt.float32).to(device)
    return rff_param_tuple(w=w_freq, b=None)


def rff_rahimi_recht(x, rff_params):
    """
    implementation more faithful to rahimi+recht paper
    """
    w = rff_params.w
    b = rff_params.b
    xwt = pt.mm(x, w.t()) + b
    z = pt.cos(xwt)
    z = z * pt.sqrt(pt.tensor(2. / w.shape[0]).to(pt.float32))
    return z


def weights_rahimi_recht(d_rff, d_enc, sig, device):
    w_freq = pt.tensor(np.random.randn(d_rff, d_enc) / np.sqrt(sig)).to(pt.float32).to(device)
    b_freq = pt.tensor(np.random.rand(d_rff) * (2 * np.pi * sig)).to(device)
    return rff_param_tuple(w=w_freq, b=b_freq)


def data_label_embedding(data, labels, rff_params, mmd_type,
                         labels_to_one_hot=False, n_labels=None, device=None, reduce='mean'):
    assert reduce in {'mean', 'sum'}
    data_embedding = rff_sphere(data, rff_params) if mmd_type == 'sphere' else rff_rahimi_recht(data, rff_params)
    if n_labels is None:
        return pt.mean(data_embedding, 0) if reduce == 'mean' else pt.sum(data_embedding, 0)
    if labels_to_one_hot:
        batch_size = data.shape[0]
        one_hots = pt.zeros(batch_size, n_labels, device=device)
        one_hots.scatter_(1, labels[:, None], 1)
        labels = one_hots
    embedding = pt.einsum('ki,kj->kij', [data_embedding, labels])
    return pt.mean(embedding, 0) if reduce == 'mean' else pt.sum(embedding, 0)


def get_rff_mmd_loss(d_enc, d_rff, rff_sigma, device, n_labels, noise_factor, batch_size, mmd_type):
    assert d_rff % 2 == 0
    d_rff_reduced = int(d_rff / 2)

    if mmd_type == 'sphere':
        w_freq = weights_sphere(d_rff, d_enc, rff_sigma, device)
    else:
        w_freq = weights_rahimi_recht(d_rff, d_enc, rff_sigma, device)

    def rff_mmd_loss(data_enc, labels, gen_enc, gen_labels, alpha=None, beta=None):
        data_emb = data_label_embedding(data_enc, labels, w_freq, mmd_type, labels_to_one_hot=True,
                                        n_labels=n_labels, device=device)
        gen_emb = data_label_embedding(gen_enc, gen_labels, w_freq, mmd_type)
        noise = pt.randn_like(data_emb) * (2 * noise_factor / batch_size)
        noisy_emb = data_emb + noise
        # return pt.sum((noisy_emb - gen_emb) ** 2)

        noisy_emb_real, noisy_emb_imag = noisy_emb[:d_rff_reduced, :], noisy_emb[d_rff_reduced:, :],
        t_target_norm = calculate_norm(noisy_emb_real, noisy_emb_imag)

        gen_emb_real, gen_emb_imag = gen_emb[:d_rff_reduced, :], noisy_emb[d_rff_reduced:, :]
        t_x_norm = calculate_norm(gen_emb_real, gen_emb_imag)

        amp_diff = t_target_norm - t_x_norm
        loss_amp = pt.mul(amp_diff, amp_diff)

        loss_pha = 2 * (pt.mul(t_target_norm, t_x_norm) -
                        pt.mul(noisy_emb_real, gen_emb_real) -
                        pt.mul(noisy_emb_imag, gen_emb_imag))

        loss_pha = loss_pha.clamp(min=1e-12)

        if alpha == None:
            alpha = 0.5
        if beta == None:
            beta = 0.5

        loss = pt.sum(alpha * loss_amp + beta * loss_pha)
        return loss

    return rff_mmd_loss, w_freq


def get_single_sigma_losses(train_loader, d_enc, d_rff, rff_sigma, device, noise_factor, mmd_type,
                            n_labels=None, pca_vecs=None):  # , alpha = None, beta = None ):
    """
      returns

      single release loss :  loss func where one side is based on noisy emb
      mb_losss: loss func where noise is divided by batch size
    """
    assert d_rff % 2 == 0
    d_rff_reduced = int(d_rff / 2)
    # w_freq = pt.tensor(np.random.randn(d_rff // 2, d_enc) / np.sqrt(rff_sigma)).to(pt.float32).to(device)
    minibatch_loss, w_freq = get_rff_mmd_loss(d_enc, d_rff, rff_sigma, device, n_labels, 0.,
                                              train_loader.batch_size, mmd_type)

    noisy_emb = noisy_dataset_embedding(train_loader, w_freq, d_rff, device, noise_factor, mmd_type,
                                        n_labels=n_labels, pca_vecs=pca_vecs)

    def single_release_loss(gen_enc, gen_labels, alpha=None, beta=None):

        gen_emb = data_label_embedding(gen_enc, gen_labels, w_freq, mmd_type)

        # return pt.sum((noisy_emb - gen_emb) ** 2)

        noisy_emb_real, noisy_emb_imag = noisy_emb[:d_rff_reduced, :], noisy_emb[d_rff_reduced:, :],
        t_target_norm = calculate_norm(noisy_emb_real, noisy_emb_imag)

        gen_emb_real, gen_emb_imag = gen_emb[:d_rff_reduced, :], noisy_emb[d_rff_reduced:, :]
        t_x_norm = calculate_norm(gen_emb_real, gen_emb_imag)

        amp_diff = t_target_norm - t_x_norm
        loss_amp = pt.mul(amp_diff, amp_diff)

        loss_pha = 2 * (pt.mul(t_target_norm, t_x_norm) -
                        pt.mul(noisy_emb_real, gen_emb_real) -
                        pt.mul(noisy_emb_imag, gen_emb_imag))

        loss_pha = loss_pha.clamp(min=1e-12)

        if alpha == None:
            alpha = 0.5
        if beta == None:
            beta = 0.5

        loss = pt.sum(alpha * loss_amp + beta * loss_pha)

        # return pt.sum((noisy_emb - gen_emb) ** 2)
        return loss

    return single_release_loss, minibatch_loss, noisy_emb, w_freq


def noisy_dataset_embedding(train_loader, w_freq, d_rff, device, noise_factor, mmd_type, sum_frequency=25,
                            n_labels=None, pca_vecs=None):
    emb_acc = []
    n_data = 0

    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        data = flat_data(data, labels, device, n_labels=10, add_label=False)

        data = data if pca_vecs is None else apply_pca(pca_vecs, data)
        emb_acc.append(data_label_embedding(data, labels, w_freq, mmd_type, labels_to_one_hot=True, n_labels=n_labels,
                                            device=device, reduce='sum'))
        # emb_acc.append(pt.sum(pt.einsum('ki,kj->kij', [rff_gauss(data, w_freq), one_hots]), 0))
        n_data += data.shape[0]

        if len(emb_acc) > sum_frequency:
            emb_acc = [pt.sum(pt.stack(emb_acc), 0)]

    print('done collecting batches, n_data', n_data)
    print(emb_acc[0].shape)
    emb_acc = pt.sum(pt.stack(emb_acc), 0) / n_data
    print(pt.norm(emb_acc), emb_acc.shape)
    noise = pt.randn_like(emb_acc) * (2 * noise_factor / n_data) * 0
    noisy_emb = emb_acc + noise
    return noisy_emb


def get_multi_sigma_minibatch_loss(d_enc, d_rff, rff_sigmas, device, n_labels, noise_factor, batch_size, mmd_type):
    w_freqs = []
    mb_losses = []
    for rff_sigma in rff_sigmas:
        mb_loss, w_freq = get_rff_mmd_loss(d_enc, d_rff, rff_sigma, device, n_labels, noise_factor, batch_size,
                                           mmd_type)
        mb_losses.append(mb_loss)
        w_freqs.append(w_freq)

    def mb_multi_loss(data_enc, labels, gen_enc, gen_labels):
        loss_acc = 0
        for loss in mb_losses:
            loss_acc += loss(data_enc, labels, gen_enc, gen_labels)
        return loss_acc

    return mb_multi_loss, w_freqs


def get_multi_sigma_losses(train_loader, d_enc, d_rff, rff_sigmas, device, n_labels, noise_factor, mmd_type):
    mb_multi_loss, w_freqs = get_multi_sigma_minibatch_loss(d_enc, d_rff, rff_sigmas, device, n_labels, noise_factor,
                                                            train_loader.batch_size, mmd_type)

    sr_losses = []
    noisy_embs = []
    for w_freq in w_freqs:
        noisy_emb = noisy_dataset_embedding(train_loader, w_freq, d_rff, device, n_labels, noise_factor, mmd_type)
        noisy_embs.append(noisy_emb)

        def single_release_loss(gen_enc, gen_labels):
            gen_emb = data_label_embedding(gen_enc, gen_labels, w_freq, mmd_type)
            return pt.sum((noisy_emb - gen_emb) ** 2)

        sr_losses.append(single_release_loss)

    def sr_multi_loss(gen_enc, gen_labels):
        loss_acc = 0
        for loss in sr_losses:
            loss_acc += loss(gen_enc, gen_labels)
        return loss_acc

    return sr_multi_loss, mb_multi_loss, noisy_embs


def get_rff_losses(train_loader, d_enc, d_rff, rff_sigma, device, noise_factor, mmd_type, n_labels=None, pca_vecs=None,
                   nested_losses=False):
    assert mmd_type in {'sphere', 'r+r'}
    assert isinstance(rff_sigma, str)
    rff_sigma = [float(sig) for sig in rff_sigma.split(',')]
    if nested_losses:
        return get_nested_losses(train_loader, d_rff, rff_sigma, device, n_labels, noise_factor, mmd_type)
    elif len(rff_sigma) == 1:
        return get_single_sigma_losses(train_loader, d_enc, d_rff, rff_sigma[0], device, noise_factor, mmd_type,
                                       n_labels, pca_vecs)
    else:
        return get_multi_sigma_losses(train_loader, d_enc, d_rff, rff_sigma, device, n_labels, noise_factor, mmd_type)


def apply_pca(sing_vecs, x):
    return x @ sing_vecs


def get_nested_losses(train_loader, d_rff, rff_sigmas, device, n_labels, noise_factor, mmd_type):
    mb_multi_loss, w_freqs = get_nested_minibatch_loss(d_rff, rff_sigmas, device, n_labels, noise_factor,
                                                       train_loader.batch_size, mmd_type)

    sr_losses = []
    noisy_embs = []
    for w_freq in w_freqs:
        noisy_emb = noisy_dataset_embedding(train_loader, w_freq, d_rff, device, n_labels, noise_factor, mmd_type)
        noisy_embs.append(noisy_emb)

        def single_release_loss(gen_enc, gen_labels):
            gen_emb = data_label_embedding(gen_enc, gen_labels, w_freq, mmd_type)
            return pt.sum((noisy_emb - gen_emb) ** 2)

        sr_losses.append(single_release_loss)

    def sr_multi_loss(gen_enc, gen_labels):
        loss_acc = 0
        for loss in sr_losses:
            loss_acc += loss(gen_enc, gen_labels)
        return loss_acc

    return sr_multi_loss, mb_multi_loss, noisy_embs


def get_nested_minibatch_loss(d_rff, rff_sigma, device, n_labels, noise_factor, batch_size, mmd_type):
    w_freqs = []

    macro_mb_loss, w_freq = get_nested_loss(28 ** 2, d_rff, rff_sigma, device, n_labels, noise_factor, batch_size,
                                            mmd_type)
    w_freqs.append(w_freq)

    micro_mb_loss, w_freq = get_nested_loss(7 ** 2, d_rff, rff_sigma, device, n_labels, noise_factor, batch_size,
                                            mmd_type)
    w_freqs.append(w_freq)

    def mb_multi_loss(data_enc, labels, gen_enc, gen_labels):
        macro_loss = macro_mb_loss(data_enc, labels, gen_enc, gen_labels)

        micro_loss = 0
        return macro_loss + micro_loss

    return mb_multi_loss, w_freqs


def get_nested_loss(d_enc, d_rff, rff_sigma, device, n_labels, noise_factor, batch_size, mmd_type):
    # alpha = None, beta = None)
    assert d_rff % 2 == 0

    if mmd_type == 'sphere':
        w_freq = weights_sphere(d_rff, d_enc, rff_sigma, device)
    else:
        w_freq = weights_rahimi_recht(d_rff, d_enc, rff_sigma, device)

    def rff_mmd_loss(data_enc, labels, gen_enc, gen_labels, alpha=None, beta=None):
        data_emb = data_label_embedding(data_enc, labels, w_freq, mmd_type, labels_to_one_hot=True,
                                        n_labels=n_labels, device=device)
        gen_emb = data_label_embedding(gen_enc, gen_labels, w_freq, mmd_type)
        noise = pt.randn(d_rff, n_labels, device=device) * (2 * noise_factor / batch_size)
        noisy_emb = data_emb + noise

        d_rff_reduced = int(d_rff // 2)

        noisy_emb_real, noisy_emb_imag = noisy_emb[:d_rff_reduced, :], noisy_emb[d_rff_reduced:, :],
        t_target_norm = calculate_norm(noisy_emb_real, noisy_emb_imag)

        gen_emb_real, gen_emb_imag = gen_emb[:d_rff_reduced, :], noisy_emb[d_rff_reduced:, :]
        t_x_norm = calculate_norm(gen_emb_real, gen_emb_imag)

        amp_diff = t_target_norm - t_x_norm
        loss_amp = pt.mul(amp_diff, amp_diff)

        loss_pha = 2 * (pt.mul(t_target_norm, t_x_norm) -
                        pt.mul(noisy_emb_real, gen_emb_real) -
                        pt.mul(noisy_emb_imag, gen_emb_imag))

        loss_pha = loss_pha.clamp(min=1e-12)

        if alpha == None:
            alpha = 0.5
        if beta == None:
            beta = 0.5

        loss = pt.sum(alpha * loss_amp + beta * loss_pha)

        # return pt.sum((noisy_emb - gen_emb) ** 2)
        return loss

    return rff_mmd_loss, w_freq
