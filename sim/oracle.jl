using StatsBase

type Oracle
    num_reporters::Int
    num_events::Int
    reports::Array{Float64,2}
    scaled::Array{Bool,1}
    scaled_max::Array{Float64,1}
    scaled_min::Array{Float64,1}
    reputation::Array{Float64,1}
    catch_tolerance::Float64
    alpha::Float64
    verbose::Bool
    algorithm::String
    variance_threshold::Float64
end

oracle = Oracle(10, rand(10,10), Dict(), rand(10), 0.1, 0.1, false, "first-component", 0.9)

function normalize{T<:Real}(v::Array{T,1})
    v = abs(v)
    if sum(v) == 0
        v += 1
    end
    vec(v) / sum(v)
end

function roundoff(x::Real, tolerance::Real)
    center = 0
    if x < center - tolerance
        return -1
    elseif x > center + tolerance
        return 1
    else
        return 0
    end
end

function interpolate(oracle::Oracle)
    reports = copy(oracle.reports)

    # Rescale scaled events
    if any(oracle.scaled)
        for i = 1:oracle.num_events
            if oracle.scaled[i]
                reports[:,i] = (reports[:,i] - scaled_min[i]) / (scaled_max[i] - scaled_min[i])
            end
        end
    end

    # Interpolation to fill missing observations
    for j = 1:oracle.num_events
        if any(isnan(reports[:,j]))
            total_active_reputation = 0
            active_reputation = (Float64)[]
            active_reports = (Float64)[]
            nan_indices = (Int)[]
            num_present = 0
            for i = 1:oracle.num_reporters
                if ~isnan(reports[i,j])
                    total_active_reputation += oracle.reputation[i]
                    push!(active_reputation, oracle.reputation[i])
                    push!(active_reports, reports[i,j])
                    num_present += 1
                else
                    push!(nan_indices, i)
                end
            end
            if !oracle.scaled[i]
                guess = 0
                for i = 1:oracle.num_present
                    active_reputation[i] /= total_active_reputation
                    guess += active_reputation[i] * active_reports[i]
                end
                guess = roundoff(guess)
            else
                for i = 1:oracle.num_present
                    active_reputation[i] /= total_active_reputation
                end
                guess = wmedian(active_reports, active_reputation)
            end
            for nan_index in nan_indices
                reports[nan_index,j] = guess
            end
        end
    end
    return reports
end

function weighted_pca{T<:Real}(reports_filled::Array{T,2})
    convergence = false
    net_adj_prin_comp = nothing

    weighted_means = []
    total_weight = 0
    i = 0
    while i < num_reporters
        j = 0
        while j < num_events
            weighted_means[j] += reputation[i] * reports_filled[i,j]
            j += 1
        end
        total_weight += reputation[i]
        i += 1
    end
    j = 0
    while j < num_events
        weighted_means[j] /= total_weight
        j += 1
    end

    weighted_centered_data = []
    i = 0
    while i < num_events
        weighted_centered_data[:,i] = reports_filled[:,i] - weighted_means[i]
        i += 1
    end

    # Compute the weighted mean (of all reporters) for each event
    weighted_mean = np.ma.average(reports_filled,
                                  axis=0,
                                  weights=self.reputation.tolist())

    # Each report's difference from the mean of its event (column)
    mean_deviation = np.matrix(reports_filled - weighted_mean)

    # Compute the unbiased weighted population covariance
    # (for uniform weights, equal to np.cov(reports_filled.T, bias=1))
    covariance_matrix = np.ma.multiply(mean_deviation.T, self.reputation).dot(mean_deviation) / float(1 - np.sum(self.reputation**2))

    # H is the un-normalized eigenvector matrix
    H = np.linalg.svd(covariance_matrix)[0]

    # Normalize loading by Euclidean distance
    first_loading = np.ma.masked_array(H[:,0] / np.sqrt(np.sum(H[:,0]**2)))
    first_score = np.dot(mean_deviation, first_loading)

    if self.algorithm == "first-component":

        set1 = first_score + np.abs(np.min(first_score))
        set2 = first_score - np.max(first_score)
        old = np.dot(self.reputation.T, reports_filled)
        new1 = np.dot(self.normalize(set1), reports_filled)
        new2 = np.dot(self.normalize(set2), reports_filled)
        ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
        adj_prin_comp = set1 if ref_ind <= 0 else set2
        net_adj_prin_comp = adj_prin_comp
        convergence = True

    elif self.algorithm == "fixed-var-length":

        U, Sigma, Vt = np.linalg.svd(covariance_matrix)
        variance_explained = np.cumsum(Sigma / np.trace(covariance_matrix))
        length = 0
        for i, var_exp in enumerate(variance_explained):
            loading = U.T[i]
            score = Sigma[i] * np.dot(mean_deviation, loading)
            length += score**2
            if var_exp > self.variance_threshold: break
        if self.verbose:
            print i, "components"
        self.num_components = i
        length = np.sqrt(length)
        set1 = length + np.abs(np.min(length))
        set2 = length - np.max(length)
        old = np.dot(self.reputation.T, reports_filled)
        new1 = np.dot(self.normalize(set1), reports_filled)
        new2 = np.dot(self.normalize(set2), reports_filled)
        ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
        net_adj_prin_comp = set1 if ref_ind <= 0 else set2
        convergence = True

    elif self.algorithm == "fixed-variance":

        U, Sigma, Vt = np.linalg.svd(covariance_matrix)
        variance_explained = np.cumsum(Sigma / np.trace(covariance_matrix))
        for i, var_exp in enumerate(variance_explained):
            loading = U.T[i]
            score = np.dot(mean_deviation, loading)
            if i == 0:
                net_score = Sigma[i] * score
            else:
                net_score += Sigma[i] * score
            if var_exp > self.variance_threshold: break
        if self.verbose:
            print i, "components"
        self.num_components = i
        set1 = net_score + np.abs(np.min(net_score))
        set2 = net_score - np.max(net_score)
        old = np.dot(self.reputation.T, reports_filled)
        new1 = np.dot(self.normalize(set1), reports_filled)
        new2 = np.dot(self.normalize(set2), reports_filled)
        ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
        net_adj_prin_comp = set1 if ref_ind <= 0 else set2
        convergence = True            

    elif self.algorithm == "ica-adjusted":
        ica = FastICA(n_components=self.num_events, whiten=False)
                      # random_state=0,
                      # max_iter=1000)
        while not convergence:
            with warnings.catch_warnings(record=True) as w:
                try:
                    S_ = ica.fit_transform(covariance_matrix)   # Reconstruct signals
                    if len(w):
                        continue
                    else:
                        if self.verbose:
                            print "ICA loadings:"
                            print S_
                        S_first_loading = S_[:,0]
                        S_first_loading /= np.sqrt(np.sum(S_first_loading**2))
                        S_first_score = np.array(np.dot(mean_deviation, S_first_loading)).ravel()

                        S_set1 = S_first_score + np.abs(np.min(S_first_score))
                        S_set2 = S_first_score - np.max(S_first_score)
                        S_old = np.dot(self.reputation.T, reports_filled)
                        S_new1 = np.dot(self.normalize(S_set1), reports_filled)
                        S_new2 = np.dot(self.normalize(S_set2), reports_filled)
                        S_ref_ind = np.sum((S_new1 - S_old)**2) - np.sum((S_new2 - S_old)**2)
                        S_adj_prin_comp = S_set1 if S_ref_ind <= 0 else S_set2
                        if self.verbose:
                            print self.normalize(S_adj_prin_comp * (self.reputation / np.mean(self.reputation)).T)
                        net_adj_prin_comp = S_adj_prin_comp
                        convergence = not any(np.isnan(net_adj_prin_comp))
                except:
                    continue

    elif self.algorithm == "ica-inverse":
        ica = FastICA(n_components=self.num_events, whiten=True)
        while not convergence:
            with warnings.catch_warnings(record=True) as w:
                try:
                    S_ = ica.fit_transform(covariance_matrix)
                    if len(w):
                        continue
                    else:
                        S_first_loading = S_[:,0]
                        S_first_loading /= np.sqrt(np.sum(S_first_loading**2))
                        S_first_score = np.array(np.dot(mean_deviation, S_first_loading)).ravel()

                        # Normalized absolute inverse scores
                        net_adj_prin_comp = 1 / np.abs(S_first_score)
                        net_adj_prin_comp /= np.sum(net_adj_prin_comp)

                        convergence = not any(np.isnan(net_adj_prin_comp))
                except:
                    continue

    elif self.algorithm == "covariance-ratio":
        # Sum over all events in the ballot; the ratio of this sum to
        # the total covariance (over all events, across all reporters)
        # is each reporter's contribution to the overall variability.

        row_mean = np.mean(reports_filled, axis=1)
        centered = np.zeros(reports_filled.shape)
        for i in range(self.num_reporters): centered[i,:] = reports_filled[i,:] - np.ones(self.num_events)*row_mean[i]
        covm = np.ma.multiply(centered, np.ones(self.num_events)).dot(centered) / float(1 - np.sum(self.reputation**2))

        # Compute the unbiased weighted population covariance
        # (for uniform weights, equal to np.cov(reports_filled.T, bias=1))
        covariance_matrix = np.ma.multiply(mean_deviation.T, self.reputation).dot(mean_deviation) / float(1 - np.sum(self.reputation**2))

        # Sum across columns of the (other) covariance matrix
        contrib = np.sum(covariance_matrix, 1)
        relative_contrib = contrib / np.sum(contrib)

        set1 = relative_contrib + np.abs(np.min(relative_contrib))
        set2 = relative_contrib - np.max(relative_contrib)
        old = np.dot(self.reputation.T, reports_filled)
        new1 = np.dot(self.normalize(set1), reports_filled)
        new2 = np.dot(self.normalize(set2), reports_filled)
        ref_ind = np.sum((new1 - old)**2) - np.sum((new2 - old)**2)
        net_adj_prin_comp = set1 if ref_ind <= 0 else set2

        convergence = True

    elif self.algorithm == "fourth-cumulant":
        if aux is not None:
            if "cokurt" in aux:
                self.cokurt = aux["cokurt"]
            if "coskew" in aux:
                self.coskew = aux["coskew"]

            # Sum over all events in the ballot; the ratio of this sum to
            # the total cokurtosis is that reporter's contribution.
            convergence = True

    elif self.algorithm == "ica-tensor":
        convergence = True
        # # Tensor-decomposition ICA (Maple -- see tensor-ICA.mw)
        # x1 := proc (t) options operator, arrow; sqrt(2)*sin(t) end proc;
        # x2 := proc (t) options operator, arrow; signum(sin(2*t)) end proc;
        # plot([x1(t), x2(t)], t = 0 .. 4*Pi, numpoints = 5000);
        # (int(x1(t)^4, t = 0 .. 4*Pi))/(4*Pi)-3*((int(x1(t)^2, t = 0 .. 4*Pi))/(4*Pi))^2;
        # (int(x2(t)^4, t = 0 .. 4*Pi))/(4*Pi)-3*((int(x2(t)^2, t = 0 .. 4*Pi))/(4*Pi))^2;
        # M := (1/4)*Matrix([[-1, -3*sqrt(3)], [3*sqrt(3), -5]]);
        # lambda, V := Eigenvectors(Transpose(M).M);
        # Sigma := map(sqrt, DiagonalMatrix(lambda));
        # v1 := Vector([V[1, 1], V[2, 1]]); v2 := Vector([V[1, 2], V[2, 2]]);
        # v1 := Normalize(v1, Euclidean); v2 := Normalize(v2, Euclidean);
        # V := Matrix([v1, v2]);
        # U := M.V.(1/Sigma);
        # U.Sigma.Transpose(V); Transpose(U).M.V;
        # y := M.Vector([x1(t), x2(t)]);
        # plot([y[1], y[2]], t = 0 .. 4*Pi);

        # # use SVD of M to expand y using the columns of U (or V`^(T)) --
        # # this is y as a linear combination of the principal components of M
        # z := simplify(1/Sigma.Transpose(U).y);
        # plot([z[1], z[2]], t = 0 .. 4*Pi);
        # (int(z[1]*z[2], t = 0 .. 4*Pi))/(4*Pi);

        # # these are uncorrelated, but clearly are not the original
        # # signals x1(t) and x2(t)
        # # - the problem is that ANY orthogonal rotation of z results in
        # #   mutually uncorrelated signals
        # # - in other words, there is a particular (unknown) orthogonal
        # #   rotation of x that gives z
        # # - to find it, set up the fourth-order supersymmetric cumulant
        # #   tensor:
        # #   E{1,2,3,4} - E{1,2}⋅E{3,4} - E{1,3}⋅E{2,4} - E{1,4}⋅E{2,3}
        # a[1, 1, 1, 1] := (int(z[1]^4, t = 0 .. 4*Pi))/(4*Pi)-3*((int(z[1]^2, t = 0 .. 4*Pi))/(4*Pi))^2;
        # a[1, 1, 1, 2] := (int(z[1]^3*z[2], t = 0 .. 4*Pi))/(4*Pi)-3*(int(z[1]^2, t = 0 .. 4*Pi))/(4*Pi)*((int(z[1]*z[2], t = 0 .. 4*Pi))/(4*Pi));
        # a[1, 1, 2, 1] := a[1, 1, 1, 2]; a[1, 2, 1, 1] := a[1, 1, 1, 2]; a[2, 1, 1, 1] := a[1, 1, 1, 2];
        # a[1, 1, 2, 2] := (int(z[1]^2*z[2]^2, t = 0 .. 4*Pi))/(4*Pi)-(int(z[1]^2, t = 0 .. 4*Pi))/(4*Pi)*((int(z[2]^2, t = 0 .. 4*Pi))/(4*Pi))-2*((int(z[1]*z[2], t = 0 .. 4*Pi))/(4*Pi))^2;
        # a[1, 2, 1, 2] := a[1, 1, 2, 2]; a[1, 2, 2, 1] := a[1, 1, 2, 2]; a[2, 1, 2, 1] := a[1, 1, 2, 2]; a[2, 2, 1, 1] := a[1, 1, 2, 2]; a[2, 1, 1, 2] := a[1, 1, 2, 2];
        # a[1, 2, 2, 2] := (int(z[1]*z[2]^3, t = 0 .. 4*Pi))/(4*Pi)-3*(int(z[2]^2, t = 0 .. 4*Pi))/(4*Pi)*((int(z[1]*z[2], t = 0 .. 4*Pi))/(4*Pi));
        # a[2, 1, 2, 2] := a[1, 2, 2, 2]; a[2, 2, 1, 2] := a[1, 2, 2, 2]; a[2, 2, 2, 1] := a[1, 2, 2, 2];
        # a[2, 2, 2, 2] := (int(z[2]^4, t = 0 .. 4*Pi))/(4*Pi)-3*((int(z[2]^2, t = 0 .. 4*Pi))/(4*Pi))^2;
        # c1 := proc (i2, i3, i4) options operator, arrow; 4*i2-6+2*i3+i4 end proc;
        # c2 := proc (i1, i3, i4) options operator, arrow; 4*i3-6+2*i4+i1 end proc;
        # c3 := proc (i1, i2, i4) options operator, arrow; 4*i4-6+2*i1+i2 end proc;
        # c4 := proc (i1, i2, i3) options operator, arrow; 4*i1-6+2*i2+i3 end proc;
        # A1 := Matrix(2, 8); A2 := Matrix(2, 8); A3 := Matrix(2, 8); A4 := Matrix(2, 8);
        # for j to 2 do
        #     for k to 2 do
        #         for l to 2 do
        #             for m to 2 do
        #                     A1[j, c1(k, l, m)] := a[j, k, l, m];
        #                     A2[k, c2(j, l, m)] := a[j, k, l, m];
        #                     A3[l, c3(j, k, m)] := a[j, k, l, m];
        #                     A4[m, c4(j, k, l)] := a[j, k, l, m]
        #             end do
        #         end do
        #     end do
        # end do;
        # lambda, Vx := Eigenvectors(A1.Transpose(A1));
        # vx1 := Vector([Vx[1, 1], Vx[2, 1]]); vx2 := Vector([Vx[1, 2], Vx[2, 2]]);
        # vx1 := Normalize(vx1, Euclidean); vx2 := Normalize(vx2, Euclidean);
        # Vnorm := Matrix([vx1, vx2]);
        # Mhat := U.Sigma.Transpose(Vnorm);
        # finalx := simplify(1/Mhat.y);
        # plot([finalx[1], finalx[2]], t = 0 .. 4*Pi, numpoints = 2000);

    # ica kurtosis threshold?
    # use covariance matrix sum-over-rows directly, rather than pca?

    row_reward_weighted = self.reputation
    if max(abs(net_adj_prin_comp)) != 0:
        row_reward_weighted = self.normalize(net_adj_prin_comp * (self.reputation / np.mean(self.reputation)).T)

    smooth_rep = self.alpha*row_reward_weighted + (1-self.alpha)*self.reputation.T

    return {
        "first_loading": first_loading,
        "old_rep": self.reputation.T,
        "this_rep": row_reward_weighted,
        "smooth_rep": smooth_rep,
        "convergence": convergence,
    }
end

function consensus()

    # Handle missing values
    reports_filled = self.interpolate(self.reports)

    # Consensus - Row Players
    # New Consensus Reward
    player_info = self.weighted_pca(reports_filled)
    adj_first_loadings = player_info['first_loading']

    # Column Players (The Event Creators)
    # Calculation of Reward for Event Authors
    # Consensus - "Who won?" Event Outcome    
    # Simple matrix multiplication ... highest information density at reporter_bonus,
    # but need EventOutcomes.Raw to get to that
    outcomes_raw = np.dot(player_info['smooth_rep'], reports_filled).squeeze()

    # Discriminate Based on Contract Type
    if self.event_bounds is not None:
        for i in range(reports_filled.shape[1]):
            # Our Current best-guess for this Scaled Event (weighted median)
            if self.event_bounds[i]["scaled"]:
                outcomes_raw[i] = weighted_median(reports_filled[:,i].data, weights=player_info["smooth_rep"].ravel().data)

    # The Outcome (Discriminate Based on Contract Type)
    outcomes_adj = []
    for i, raw in enumerate(outcomes_raw):
        if self.event_bounds is not None and self.event_bounds[i]["scaled"]:
            outcomes_adj.append(raw)
        else:
            outcomes_adj.append(self.catch(raw))

    outcomes_final = []
    for i, raw in enumerate(outcomes_raw):
        outcomes_final.append(outcomes_adj[i])
        if self.event_bounds is not None and self.event_bounds[i]["scaled"]:
            outcomes_final[i] *= self.event_bounds[i]["max"] - self.event_bounds[i]["min"]
            outcomes_final[i] += self.event_bounds[i]["min"]

    certainty = []
    for i, adj in enumerate(outcomes_adj):
        certainty.append(sum(player_info["smooth_rep"][reports_filled[:,i] == adj]))

    certainty = np.array(certainty)
    consensus_reward = self.normalize(certainty)
    avg_certainty = np.mean(certainty)

    # Participation
    # Information about missing values
    na_mat = self.reports * 0
    na_mat[na_mat.mask] = 1  # indicator matrix for missing

    # Participation Within Events (Columns)
    # % of reputation that answered each Event
    participation_columns = 1 - np.dot(player_info['smooth_rep'], na_mat)

    # Participation Within Agents (Rows)
    # Many options
    # 1- Democracy Option - all Events treated equally.
    participation_rows = 1 - na_mat.sum(axis=1) / na_mat.shape[1]

    # General Participation
    percent_na = 1 - np.mean(participation_columns)

    # Possibly integrate two functions of participation? Chicken and egg problem...
    if self.verbose:
        print('*Participation Information*')
        print('Voter Turnout by question')
        print(participation_columns)
        print('Voter Turnout across questions')
        print(participation_rows)

    # Combine Information
    # Row
    na_bonus_reporters = self.normalize(participation_rows)
    reporter_bonus = na_bonus_reporters * percent_na + player_info['smooth_rep'] * (1 - percent_na)

    # Column
    na_bonus_events = self.normalize(participation_columns)
    author_bonus = na_bonus_events * percent_na + consensus_reward * (1 - percent_na)

    return {
        'original': self.reports.data,
        'filled': reports_filled.data,
        'agents': {
            'old_rep': player_info['old_rep'],
            'this_rep': player_info['this_rep'],
            'smooth_rep': player_info['smooth_rep'],
            'na_row': na_mat.sum(axis=1).data.tolist(),
            'participation_rows': participation_rows.data.tolist(),
            'relative_part': na_bonus_reporters.data.tolist(),
            'reporter_bonus': reporter_bonus.data.tolist(),
            },
        'events': {
            'adj_first_loadings': adj_first_loadings.data.tolist(),
            'outcomes_raw': outcomes_raw.data.tolist(),
            'consensus_reward': consensus_reward,
            'certainty': certainty,
            'NAs Filled': na_mat.sum(axis=0).data.tolist(),
            'participation_columns': participation_columns.data.tolist(),
            'author_bonus': author_bonus.data.tolist(),
            'outcomes_adjusted': outcomes_adj,
            'outcomes_final': outcomes_final,
            },
        'participation': 1 - percent_na,
        'avg_certainty': avg_certainty,
        'convergence': player_info['convergence'],
        'components': self.num_components,
    }

end
