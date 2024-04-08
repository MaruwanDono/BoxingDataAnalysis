import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer

#Cluster indices for bootstrap
cl1_indices = []
cl2_indices = []

def preprocessing():
    try:
        path = "Datasets/Boxing/"
        file_name = 'boxing_matches.csv'
        df = pd.read_csv(path+file_name)
        ignore_columns = ['judge1_A','judge1_B','judge2_A','judge2_B','judge3_A','judge3_B']
        df.dropna(subset=df.columns.difference(ignore_columns), inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['result'] = df['result'].replace('win_A','win').replace('win_B', 'loss')
        #Categories
        #print(df['result'].value_counts())
        cols = df.columns.tolist()
        for i,col in enumerate(cols):
            if ('_B' in col) and (df[col].dtype.kind in 'iufc'):
                df[col] = df[cols[i-1]] - df[col]
        new_cols = {col:col.replace('_A','').replace('_B','_diff') for col in cols}
        df.rename(columns=new_cols, inplace=True)
        #Reach of more than 4m isnt acceptable, drop rows
        df.drop(labels=[2805,2963,2893], axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(path+file_name.replace('.csv', '_updated.csv'), index=False)
        return df
    except Exception as e:
        print(r'Unable to read data: {}'.format(str(e)))
        return pd.DataFrame()


def correlation(df):
    try:
        print('Correlation:')
        #plt.matshow(df.corr(numeric_only=True))
        #plt.show()
        #pd.plotting.scatter_matrix(df, alpha=0.5, figsize=(6, 6), diagonal='hist')
        X = df['height'].to_numpy()
        Y = df['reach'].to_numpy()
        #Reach of more than 4m is irrational, drop those rows
        #Rho from library
        rho = pearsonr(X,Y).statistic
        #Rho built with formula
        rho_ = sum([(X[i]-np.mean(X))*(Y[i]-np.mean(Y)) for i in range(len(X))])/(len(X)*np.std(X)*np.std(Y))
        print('Library rho = {}, calculated rho={}'.format(rho,rho_))
        a = rho*np.std(Y)/np.std(X)
        b = np.mean(Y) - a*np.mean(X)
        print('a={}, b={}'.format(a,b))
        print('Correlation coeff: {}, determinancy coeff: {}'.format(rho,rho**2))
        #Scatter-plot
        plt.plot(X, [a*x+b for x in X], color='crimson', label='Linear regression')
        plt.scatter(X, Y, color='blue', label='Data points')
        plt.xlabel('Height (in cm)')
        plt.ylabel('Reach (in cm)')
        plt.legend(loc='upper left')
        plt.title('Scatter-plot of height and reach for boxers')
        plt.show()
        #Compute errors
        labels = [i for i in range(len(X))]
        error_DS = [100*abs(a*x+b-Y[i])/abs(Y[i]) for i, x in enumerate(X)]
        error_ML = [100*abs(a*x+b-Y[i])/abs(a*x+b) for i, x in enumerate(X)]
        plt.plot(labels, error_DS, color='crimson', label='Data Analysis Error')
        plt.plot(labels, error_ML, color='darkblue', label='Machine Learning Error')
        plt.xlabel('Objects')
        plt.ylabel('Error (in %)')
        plt.legend(loc='upper right')
        plt.title('Regression errors')
        plt.show()
    except Exception as e:
        print(r'Unable to perform linear regression: {}'.format(str(e)))



def PCA(df):
    try:
        print('\n\nPrincipal component analysis:')
        features = ['age','age_diff','won','won_diff','lost','lost_diff']
        PCA_df_z = pd.DataFrame()
        PCA_df_r = pd.DataFrame()
        HiddenFactor_df = pd.DataFrame()

        for feature in features:
            #Standardization: z-scoring
            sigma = np.std(df[feature])
            mean = np.mean(df[feature])
            max = df[feature].max()
            min = df[feature].min()
            PCA_df_z[feature] = df[feature].apply(lambda x: (x-mean)/sigma)
            PCA_df_r[feature] = df[feature].apply(lambda x: (x-mean)/(max-min))
            HiddenFactor_df[feature] = df[feature].apply(lambda x: 100*(x-min)/(max-min))


        #Visualisation indices
        df_inf = df.loc[df['age']>=40]
        df_sup = df.loc[df['age']<=25]
        #print(df_inf.index.values.tolist())
        inf_indices = df_inf.index.values.tolist()
        sup_indices = df_sup.index.values.tolist()
        med_indices = [i for i in range(len(df)) if i not in inf_indices + sup_indices]

        ######################################
        ########### PCA z-scoring ############
        ######################################

        print('z-scoring')
        X_z = PCA_df_z.to_numpy()
        Z_z, mu_z, C_z = np.linalg.svd(X_z, full_matrices=True)
        print(np.around(Z_z,decimals=3))
        print('\n')
        print(np.around(mu_z,decimals=3))
        print('\n')
        print(np.around(C_z,decimals=3))
        #Data scatter
        tm_z= sum([mu_i**2 for mu_i in mu_z])
        t_z= np.sum(X_z**2)
        print('\nData scatter with z-scoring: t_mu={}, t={}'.format(tm_z,t_z))
        print('mu_z in percentage', [round(100*mu_i**2/t_z, 2) for mu_i in mu_z])
        Z_1 = Z_z[:,0]
        Z_2 = -Z_z[:,1]
        plt.scatter(Z_1[inf_indices], Z_2[inf_indices], color='red', label="age <= 25")
        plt.scatter(Z_1[sup_indices], Z_2[sup_indices], color='green', label="age >= 40")
        plt.scatter(Z_1[med_indices], Z_2[med_indices], color='darkblue')
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.legend(loc='upper right')
        plt.title('PCA with z-scoring')
        plt.show()

        ######################################
        ###### PCA Range Normalisation #######
        ######################################

        print('\n\nRange normalisation')
        X_r = PCA_df_r.to_numpy()
        Z_r, mu_r, C_r = np.linalg.svd(X_r, full_matrices=True)
        print(np.around(Z_r,decimals=3))
        print('\n')
        print(np.around(mu_r,decimals=3))
        print('\n')
        print(np.around(C_r,decimals=3))
        #Data scatter
        tm_r = sum([mu_i**2 for mu_i in mu_r])
        t_r = np.sum(X_r**2)
        print('\nData scatter with range normalisation: t_mu={}, t={}'.format(tm_r,t_r))
        print('mu_r in percentage', [round(100*mu_i**2/t_r,2) for mu_i in mu_r])
        Z_1 = -Z_r[:,0]
        Z_2 = -Z_r[:,1]
        plt.scatter(Z_1[inf_indices], Z_2[inf_indices], color='red', label="age <= 25")
        plt.scatter(Z_1[sup_indices], Z_2[sup_indices], color='green', label="age >= 40")
        plt.scatter(Z_1[med_indices], Z_2[med_indices], color='darkblue')
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        plt.legend(loc='upper right')
        plt.title('PCA with range normalisation')
        plt.show()

        ######################################
        ########### Hidden Factor ############
        ######################################
        X_h = HiddenFactor_df.to_numpy()
        Z_h, mu_h, C_h = np.linalg.svd(X_h, full_matrices=True)
        print(np.around(C_h, decimals=3))
        C_1 = -C_h[:,0]
        alpha = 1/sum(C_1)
        print(alpha)
    except Exception as e:
        print(r'Unable to apply PCA: {}'.format(str(e)))


def KMeansClustering(df):
    try:
        print('\n\nKMeans Clustering:')
        features = ['age','age_diff','reach','reach_diff','won','won_diff','kos','kos_diff']
        KMeans_df = pd.DataFrame()
        global_means = {}
        #Standardisation of the data
        for feature in features:
            #Standardization: z-scoring
            sigma = np.std(df[feature])
            mean = np.mean(df[feature])
            global_means[feature] = round(mean,2)
            max = df[feature].max()
            min = df[feature].min()
            KMeans_df[feature] = df[feature].apply(lambda x: (x-mean)/(max-min))
            #KMeans_z_df[feature] = df[feature].apply(lambda x: 100*(x-mean)/(sigma))

        #Clustering
        X = KMeans_df.to_numpy()
        #X_z = KMeans_z_df
        K_list = [4,7]
        n_iterations = 10
        best_iterations = {}

        for K in K_list:
            kmeans = KMeans(n_clusters=K,init='random', n_init=10)
            inertia = []
            centers = []
            labels = []
            index_best = 0
            avg_rel_errs = []
            for i in range(n_iterations):
                kmeans.fit(X)
                inertia.append(kmeans.inertia_)
                centers.append(kmeans.cluster_centers_)
                labels.append(kmeans.labels_)
                if inertia[i]<inertia[index_best]:
                    index_best = i

            #Plot inertia
            plt.plot(range(10), inertia, marker='o')
            plt.xlabel('Iterations')
            plt.ylabel('Inertia')
            plt.title('Inertia over different iterations for K={}'.format(K))
            plt.show()
            #plot clusters
            #Age for x axis
            x_feat = 4
            #Number of wins for y axis
            y_feat = 2
            #Normalised centers
            centers_ = centers[index_best]
            #Real data
            X_real = df[features].to_numpy()
            #Real centers
            centers_real = np.empty(shape=(K,len(features)))
            for j, feature in enumerate(features):
                mean = np.mean(df[feature])
                max = df[feature].max()
                min = df[feature].min()
                for i in range(K):
                    centers_real[i,j] = centers_[i,j]*(max-min)+mean

            #Visualisation
            #Normalised data
            #plt.scatter(X[:,x_feat],X[:,y_feat],c=labels[index_best],cmap='viridis',s=50,alpha=0.8)
            #plt.scatter(centers_[:,x_feat],centers_[:,y_feat],marker='X',color='red',s=200,label='Centers')
            #plt.xlabel('age')
            #plt.ylabel('wins')
            #plt.legend()
            #plt.title('KMeans clustering, K={}'.format(K))
            #plt.show()

            #Real data
            plt.scatter(X_real[:,x_feat],X_real[:,y_feat],c=labels[index_best],cmap='viridis',s=50,alpha=0.8)
            plt.scatter(centers_real[:,x_feat],centers_real[:,y_feat],marker='X',color='red',s=200,label='Centers')
            plt.xlabel('Wins')
            plt.ylabel('Reach (cm)')
            plt.legend()
            plt.title('KMeans clustering, K={}'.format(K))
            plt.show()

            #Compare Cluster means
            #Get separate clusters
            print('Global means:')
            print(global_means)
            for j in range(K):
                means = {}
                for feature in features:
                    #Standardization: z-scoring
                    mean = np.mean(df[feature])
                    max = df[feature].max()
                    min = df[feature].min()
                    means[feature] = round(np.mean(df[feature].iloc[labels[index_best] == j]),2)

                    #print(df[feature].iloc[labels[index_best] == j])
                cl_size = len(df.iloc[labels[index_best] == j])
                print('Means for cluster number {} (size {}) for K={}:'.format(j+1,cl_size,K))
                print(means)
                relative_err = [round(100*abs(means[feature]-global_means[feature])/abs(means[feature]),2) for feature in features]
                print('Relative differences to global means (%): ')
                print(relative_err)
                print('Average (%): ', round(sum(relative_err)/len(relative_err),2))
                avg_rel_errs.append(sum(relative_err)/len(relative_err))
            #Choose clusters for Bootstrap
            if (K==4):
                global cl1_indices
                cl1_indices = labels[index_best] == np.argsort(avg_rel_errs)[0]
                global cl2_indices
                cl2_indices = labels[index_best] == np.argsort(avg_rel_errs)[1]
    except Exception as e:
        print(r'Unable to read data: {}'.format(str(e)))


def ContingencyTable(df):
    try:
        print('\n\nContingency table:')
        features = ['won','won_diff']
        N_Bins = 4
        N = len(df)
        Kbins_discret = KBinsDiscretizer(n_bins=N_Bins,encode='ordinal', strategy='quantile')
        binned_data = Kbins_discret.fit_transform(df[features])
        df_binned = pd.DataFrame(binned_data, columns=features)
        #Print bins interval ranges
        for feature in features:
            bins_indices  = [df_binned[df_binned[feature]==x].index.tolist() for x in [0.0,1.0,2.0,3.0]]
            print('For feature {}, we have:'.format(feature))
            for i, indices in enumerate(bins_indices):
                print('Bin number {}: [{}, {}]'.format(i+1,df[feature].iloc[indices].min(),df[feature].iloc[indices].max()))
        print('\n')
        #Contingency tables
        c_table = pd.crosstab(df_binned[features[0]], df_binned[features[1]])
        print(c_table)
        X = c_table.to_numpy()
        CT1, CT2 = [], []
        for i in range(N_Bins):
            CT1.append(np.sum(X[:,i]))
            CT2.append(np.sum(X[i,:]))
        #Sum over columns
        sum_cols = np.array(CT1)
        #Sum over rows
        sum_rows = np.array(CT2)
        SumRowsMat = np.transpose(np.array([CT2,CT2,CT2,CT2]))
        #print(SumRowsMat)

        #Conditional probability
        print('Conditional frequency table:')
        cp_table = np.divide(X,SumRowsMat)
        #cp_table = np.divide(X,N)
        print(np.round(cp_table, decimals=2))
        #Quetelet index table and Pearson's Chi-squared
        print('Quetelet index table:')
        q_mat = np.empty(shape=(4,4))
        pearson_mat = np.empty(shape=(4,4))
        for i in range(N_Bins):
            for j in range(N_Bins):
                q_mat[i,j] = (N*X[i,j])/(sum_rows[i]*sum_cols[j])-1
                pearson_mat[i,j] = (X[i,j]-sum_rows[i]*sum_cols[j]/N)**2/(sum_rows[i]*sum_cols[j])
        print(np.round(q_mat,2))
        print('Average Quetelet index: ', round(np.sum(q_mat*np.divide(X,N)),2))
        print('Chi-squared: ', round(np.sum(pearson_mat),2))
        print('Degrees of freedom: ', 9)
        print('Pearson Chi-squared: ', round(N*np.sum(pearson_mat),2))

    except Exception as e:
        print(r'Unable to perform contingency table calculations: {}'.format(str(e)))


def Bootstrap(df):
    try:
        print('\n\nBootstrap method:')
        N_iter = 5000
        N = len(df)
        feature = 'won'
        cl1 = df.iloc[cl1_indices]
        cl2 = df.iloc[cl2_indices]
        cl1_idx = cl1.index.tolist()
        cl2_idx = cl2.index.tolist()
        indices = np.empty(shape=(N_iter,N), dtype=int)
        means = []
        means_cl1 = []
        means_cl2 = []
        for i in range(N_iter):
            indices[i,:] = np.random.randint(0, N, size=N, dtype=int)
            means.append(np.mean(df[feature].iloc[indices[i,:]]))
            L1 = []
            L2 = []
            for index in indices[i,:].tolist():
                if(cl1_indices[index]):
                    L1.append(df[feature].iloc[index])
                if(cl2_indices[index]):
                    L2.append(df[feature].iloc[index])

            if(L1):
                means_cl1.append(sum(L1)/len(L1))
            else:
                means_cl1.append(0)
            if(L2):
                means_cl2.append(sum(L2)/len(L2))
            else:
                means_cl2.append(0)

        #Confidence intervals
        print('Confidence interval for the grand mean:')
        #Pivotal
        lp = np.mean(means)-1.96*np.std(means)
        rp = np.mean(means)+1.96*np.std(means)
        print('Pivotal Ic=[{}, {}]'.format(lp,rp))
        #Non pivotal
        sorted_means = means[:]
        sorted_means.sort()
        lnp = sorted_means[125]
        rnp = sorted_means[4874]
        print('Non-pivotal Ic=[{}, {}]'.format(lnp,rnp))
        #Grand mean
        print('Grand mean: ',np.mean(df[feature]))

        #Compare cl1 and cl2 means
        #Pivotal
        m12 = [means_cl1[i] - means_cl2[i] for i in range(N_iter)]
        print('Comparing cl1 and cl2 means:')
        lp = np.mean(m12)-1.96*np.std(m12)
        rp = np.mean(m12)+1.96*np.std(m12)
        print('Pivotal Ic=[{}, {}]'.format(lp,rp))
        #Non pivotal
        sorted_means = m12[:]
        sorted_means.sort()
        lnp = sorted_means[125]
        rnp = sorted_means[4874]
        print('Non-pivotal Ic=[{}, {}]'.format(lnp,rnp))

        #Compare cl1 and grand mean
        #Pivotal
        m1g = [means[i] - means_cl1[i] for i in range(N_iter)]
        print('Comparing cl1 and grand mean:')
        lp = np.mean(m1g)-1.96*np.std(m1g)
        rp = np.mean(m1g)+1.96*np.std(m1g)
        print('Pivotal Ic=[{}, {}]'.format(lp,rp))
        #Non pivotal
        sorted_means = m1g[:]
        sorted_means.sort()
        lnp = sorted_means[125]
        rnp = sorted_means[4874]
        print('Non-pivotal Ic=[{}, {}]'.format(lnp,rnp))

        #Compare cl1 and cl2 means
        #Pivotal
        m2g = [means[i] - means_cl2[i] for i in range(N_iter)]
        print('Comparing cl2 and grand mean:')
        lp = np.mean(m2g)-1.96*np.std(m2g)
        rp = np.mean(m2g)+1.96*np.std(m2g)
        print('Pivotal Ic=[{}, {}]'.format(lp,rp))
        #Non pivotal
        sorted_means = m2g[:]
        sorted_means.sort()
        lnp = sorted_means[125]
        rnp = sorted_means[4874]
        print('Non-pivotal Ic=[{}, {}]'.format(lnp,rnp))
    except Exception as e:
        print(r'Unable to perform Bootstrap: {}'.format(str(e)))


if __name__=='__main__':
    df = preprocessing()
    correlation(df)
    PCA(df)
    KMeansClustering(df)
    ContingencyTable(df)
    Bootstrap(df)
