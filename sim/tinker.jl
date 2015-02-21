# B: vote (report) ballot
B = [[ 1.0  1.0 -1.0 -1.0],
     [ 1.0 -1.0 -1.0 -1.0],
     [ 1.0  1.0 -1.0 -1.0],
     [ 1.0  1.0  1.0 -1.0],
     [-1.0 -1.0  1.0  1.0],
     [-1.0 -1.0  1.0  1.0]]

# r: reputation vector
r = [ 0.076923,  0.384615,  0.153846,  0.076923,  0.269231,  0.038462]

# X: centered ballot
# (B'*r is the weighted mean vector)
X = B .- (B'*r)'
Σ = X'.*r'*X / (1-sum(r.^2))
