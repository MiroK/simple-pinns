import torch


def sample_hypercube(dim, npts, fixed=None, dtype=torch.float64):
    '''Batch x '''
    assert dim >= 1
    
    if fixed is None:
        return torch.rand(1, npts, dim, dtype=dtype)

    fixed_indices = []
    for idx, _ in fixed:
        if fixed_indices:
            assert idx not in fixed_indices
        fixed_indices.append(idx)

    randomized = sample_hypercube(dim=dim-len(fixed),
                                  npts=npts,
                                  fixed=None,
                                  dtype=dtype).squeeze(0)

    npts = randomized.shape[0]
    ones = torch.ones(npts, dtype=dtype)
    full = torch.zeros(npts, dim, dtype=dtype)
    for axis, val in fixed:
        full[..., axis] = val*ones

    randomized_indices = sorted(set(range(dim)) - set(fixed_indices))

    #if len(randomized_indices) == 1:
    #    randomized_indices, = randomized_indices
    full[..., randomized_indices] = randomized

    return full.unsqueeze(0)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    print(sample_hypercube(dim=2, npts=10))
    print(sample_hypercube(dim=3, npts=10, fixed=[(0, 1)]))    
                    

    

    
