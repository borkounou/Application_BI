from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np 

def dimensionnalite_reduction(X_norm):

    acp = PCA(svd_solver='full')
    coord = acp.fit_transform(X_norm)
    # nb of computed components
    print(acp.n_components_) 
    # explained variance scores
    print(acp.explained_variance_ratio_)

    # explained variance scores
    exp_var_pca = acp.explained_variance_ratio_
    print(exp_var_pca)
    # Cumulative sum of explained variance values; This will be used to
    # create step plot for visualizing the variance explained by each
    # principal component.
    #
    cum_sum_expl_var = np.cumsum(exp_var_pca)

    #
    # Create the visualization plot
    #
    fig = plt.figure()
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_expl_var)), cum_sum_expl_var, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('acp_expl_var')
    plt.close(fig)


    # plot eigen values
    n = np.size(X_norm, 0)
    p = np.size(X_norm, 1)
    eigval = float(n-1)/n*acp.explained_variance_
    fig = plt.figure()
    plt.plot(np.arange(1,p+1),eigval)
    plt.title("Screen plot")
    plt.ylabel("Eigen values")
    plt.xlabel("Factor number")
    plt.savefig('acp_eigen_values')
    plt.close(fig)

    # print eigen vectors
    print(acp.components_)
    # lines: factors
    # columns: variables

    # print correlations between factors and original variables
    sqrt_eigval = np.sqrt(eigval)
    corvar = np.zeros((p,p))
    for k in range(p):
        corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]
    print(corvar)
    # lines: variables
    # columns: factors


    # plot instances on the first plan (first 2 factors)
    fig, axes = plt.subplots(figsize=(12,12))
    axes.set_xlim(-10,10)
    axes.set_ylim(-10,10)
    for i in range(n):
        plt.annotate(y.values[i],(coord[i,0],coord[i,1]))
    plt.plot([-10,10],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-10,10],color='silver',linestyle='-',linewidth=1)
    plt.savefig('acp_instances_1st_plan')
    plt.close(fig)
