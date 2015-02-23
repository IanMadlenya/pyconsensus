centering(n::Int) = eye(n) - ones(n) * ones(n)' / n

centering{T<:Real}(n::Int, w::Vector{T}) = eye(n) - ones(n) * w'

normalize{T<:Real}(v::Vector{T}) = vec(v) / sum(v)

# B: vote (report) ballot
B = [[ 1.0  1.0 -1.0 -1.0],
     [ 1.0 -1.0 -1.0 -1.0],
     [ 1.0  1.0 -1.0 -1.0],
     [ 1.0  1.0  1.0 -1.0],
     [-1.0 -1.0  1.0  1.0],
     [-1.0 -1.0  1.0  1.0]]

num_users, num_events = size(B)

# r: reputation vector
ϱ = [2, 10, 4, 2, 7, 1]
r = normalize(ϱ)

# Per-event covariance matrix
# X: centered ballot
# Notes:
#   - B'*r is the weighted mean vector
#   - if r = [1 1 ... 1] then sum(r^2) = num_users and X.*r = X
#   - reduces to
#       C = B .- mean(B,1)
#       Σ = C' * C / num_users for unweighted
# Centering matrix (centered columns):
#   B .- mean(B,1) == centering(size(B,1)) * B
X_event = B .- (B' * r)'
Σ_event = X_event' * (X_event .* r) / (1 - sum(r.^2))

# Per-user covariance matrix
# Notes:
#   - C*C'/num_events is the unweighted covariance matrix
# Centering matrix (centered rows):
#   B .- mean(B,2) == B * centering(size(B,2))
C = B .- mean(B,2)
Σ = C*(C.*r)' / (1-sum(r.^2))
contrib = sum(Σ,2)
relative_contrib = contrib / sum(contrib)

# Brute force scoring: replicated rows
B_tmp = Array{Float64,1}[]
for i = 1:num_users
    for j = 1:ϱ[i]
        push!(B_tmp, vec(B[i,:]))
    end
end
num_rows = length(B_tmp)
B_rpl = zeros(num_rows, num_events)
for i = 1:num_rows
    for j = 1:num_events
        B_rpl[i,j] = B_tmp[i][j]
    end
end
C_rpl = B_rpl .- mean(B_rpl,2)
Σ_rpl = C_rpl * C_rpl' / num_events
contrib_rpl = sum(Σ_rpl,2)
relative_contrib_rpl = contrib_rpl / sum(contrib_rpl)
per_user_relative_contrib_rpl = zeros(num_users)
row = 1
for i = 1:num_users
    per_user_relative_contrib_rpl[i] = ϱ[i] * relative_contrib_rpl[row]
    row += ϱ[i]
end
