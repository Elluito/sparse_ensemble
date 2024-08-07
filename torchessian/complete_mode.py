import torch
import tqdm
from . import hessian_matmul


def lanczos(model, loss_function, dataloader, m, max_samples=0, buffer=2):
    """
        This in an implementation of the Lanczos Algorithm as stated in
        https://en.wikipedia.org/wiki/Lanczos_algorithm.
        Inputs:
            model: a torch.nn.Module with some parameters having requires_grad
                set True.
            loss_function: the loss the neural net is optimizing, and for which
                the hessian spectrum will be calculated.
            dataloader: a torch dataloader for which the spectrum will be
                estimated.
            m: the number of iterations of the Lanczos Algorithm.
            buffer: the number of base vectors you wish to keep in GPU memory,
                if ever you are using GPU. Set buffer to 2 if you are having
                OOM errors.
        Outputs:
            T: the tridiagonal matrix of the Lanczos Algorithm.
            V : the orthonormal basis of the Lanczos Algorithm.
    """
    n = sum(p.data.numel() for p in model.parameters() if p.requires_grad)

    assert n >= m
    assert buffer >= 2

    v = torch.ones(n)
    v /= torch.norm(v)

    w = torch.zeros_like(v)

    print("[Complete LANCZOS Algorithm running]")
    k = len(dataloader)
    device = next(model.parameters()).data.device

    counter_of_samples = 0
    for i, batch in enumerate(dataloader):
        v_ = v.to(device)
        w = w.to(device)
        size_batch = len(batch)
        batch = map(lambda x: x.to(device), batch)
        w += hessian_matmul(model, loss_function, v_, batch) / k
        counter_of_samples += size_batch
        if max_samples != 0 and counter_of_samples > max_samples:
            break

    v = v.to(w.device)
    alpha = []
    alpha.append(w.dot(v))
    w -= alpha[0] * v

    V = [v]
    beta = []

    for i in tqdm.tqdm(range(1, m)):
        b = torch.norm(w)
        beta.append(b)
        if b > 0:
            v = w / b
        else:
            done = False
            k = 0
            while not done:
                k += 1
                v = torch.rand(n).to(w.device)

                for v_ in V:
                    v_ = v_.to(v.device)
                    v -= v.dot(v_) * v_

                done = torch.norm(v) > 0
                if k > 2 and not done:  # This shouldn't happen even twice
                    raise Exception("Can't find orthogonal vector")

        # Full re-orthogonalization
        for v_ in V:
            v_ = v_.to(v.device)
            v -= v.dot(v_) * v_

        v /= torch.norm(v)
        V.append(v)

        # Saving GPU memory
        if len(V) > buffer:
            V[-buffer - 1] = V[-buffer - 1].cpu()

        w = torch.zeros_like(v)
        counter_of_samples = 0
        for j, batch in enumerate(dataloader):
            v_ = v.to(device)
            w = w.to(device)
            counter_of_samples += len(batch)
            batch = map(lambda x: x.to(device), batch)
            w += hessian_matmul(model, loss_function, v_, batch) / k
            if max_samples != 0 and counter_of_samples > max_samples:
                break

        alpha.append(w.dot(v))
        w = w - alpha[-1] * V[-1] - beta[-1] * V[-2]

    T = torch.diag(torch.Tensor(alpha))
    for i in range(m - 1):
        T[i, i + 1] = beta[i]
        T[i + 1, i] = beta[i]

    V = torch.cat(tuple(v.cpu().unsqueeze(0) for v in V), 0)
    return T, V


def gauss_quadrature(model, loss_function, dataloader, m, buffer=2):
    T, _ = lanczos(model, loss_function, dataloader, m, buffer=buffer)
    D, U = torch.linalg.eig(T)
    print(D)
    print(U)
    # L = D[:, 0]  # All eingenvalues are real
    # W = torch.Tensor(list(U[0, i] ** 2 for i in range(m)))
    return D, U
