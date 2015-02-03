players = importdata('50x25.players');
reports = importdata('50x25.dat');

liars = find(strcmp(players, 'liar'));

old_rep = zeros(50,1) + 0.02;
reputation = importdata('50x25.rep');

delta_rep = reputation - old_rep;

% covrep = cov(reports);

% categories = cell(25,1);
% categories{1} = 'Degree';
% categories{2} = 'Clustering coefficient';
% categories{3} = 'Betweenness';
% categories{4} = 'Closeness';
% categories{5} = 'Eigenvalue';
% categories{6} = 'Neighbor degree';

[pc scores latent] = princomp(zscore(reports));
percent_explained = 100 * latent / sum(latent);

figure(1)
clf
pareto(percent_explained)
xlabel('Principal component')
ylabel('Variance explained (%)')

figure(2)
clf
biplot(pc(:,1:2),'scores',scores(:,1:2));

figure(3)
clf
plot(scores(:,1),scores(:,2),'+')
xlabel('Component 1')
ylabel('Component 2')

% figure(4)
% clf
% biplot(pc(:,1:3), 'scores',scores(:,1:3));
% axis([-0.4 0.8 -0.2 0.8 -0.1 1]);
% view([30 40]);

% scores on pc 1 vs rep before/after bars
% [counts,edges] = hist(scores(:,1),length(unique(scores(:,1))));
fs = scores(:,1);
fsr = round(fs*1000)/1000;
ufsr = unique(fsr);
aufsr = abs(ufsr);
counts = zeros(length(ufsr),1);
for ii = 1:length(ufsr)
    counts(ii) = sum(fsr == ufsr(ii));
end

figure(5)
clf
plot(abs(fsr), delta_rep, '+')
hold on
for ii = 1:length(fs)
    
    % marker size proportional to number of people with that score
    csize = 10*counts(find(ufsr == fsr(ii)));

    % colors: red if liar, blue otherwise
    if find(liars == ii)
        plot(abs(fs(ii)), delta_rep(ii), 'r+')
        plot(abs(fs(ii)), delta_rep(ii), 'ro', 'MarkerSize', csize)
    else
        plot(abs(fs(ii)), delta_rep(ii), 'bo', 'MarkerSize', csize)
    end
end
xlabel('principal component')
ylabel('change in reputation')
