import numpy as np

I, J, P, Q, NU = 16, 16, 256, 256, 10

# Create fake weights
_w = np.arange(P)
_W = np.outer(_w, np.conj(_w))
w = np.zeros((I, J, P, NU))
W = np.zeros((I, J, P, Q, NU))


# Weights matrix
for ii in range(I):
    for jj in range(J):
        for nn in range(NU):
            w[ii, jj, :, nn] = _w
            W[ii, jj, :, :,nn] = _W

# Create visibility matrix
V = np.arange(P*Q*NU).reshape((P, Q, NU))

def compute_B_v1(I, J, NU):
    B = np.zeros(shape=(I, J, NU))
    for nn in range(NU):
        for ii in range(I):
            for jj in range(J):
                _V = V[:, :, nn]
                _W = W[ii, jj, :, :, nn]
                es = np.einsum('pq,pq', _W, _V)
                B[ii, jj, nn] = es
    return B

def compute_B_v2(NU):        
    B2 = np.zeros(shape=(I, J, NU))
    for nn in range(NU):
        _V = V[..., nn]
        _W = W[..., nn]
        B2[..., nn] = np.einsum('pq,ijpq->ij', _V, _W)
    return B2

def compute_B_v3():       
    B = np.einsum('ijpqn,pqn->ijn', W, V)
    return B

def compute_B_v4():
    B = np.einsum('ijpn,pqn,ijqn->ijn', w, V, np.conj(w))
    return B

B1 = compute_B_v1(I, J, NU)
B2 = compute_B_v2(NU)
B3 = compute_B_v3()
B4 = compute_B_v4()

assert np.allclose(B1, B2)
assert np.allclose(B1, B3)
assert np.allclose(B1, B4)