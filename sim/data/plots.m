players = importdata('50x25.players');
reports = importdata('50x25.dat');

covrep = cov(reports);

categories = cell(25,1);
categories{1} = 'Degree';
categories{2} = 'Clustering coefficient';
categories{3} = 'Betweenness';
categories{4} = 'Closeness';
categories{5} = 'Eigenvalue';
categories{6} = 'Neighbor degree';

[pc scores latent] = princomp(zscore(reports));
percent_explained = 100*latent/sum(latent);

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

%subplot(1,2,2);
figure(4)
clf
biplot(pc(:,1:3), 'scores',scores(:,1:3));
axis([-0.4 0.8 -0.2 0.8 -0.1 1]);
view([30 40]);
