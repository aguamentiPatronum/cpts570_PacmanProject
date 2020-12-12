import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import networkx as nx
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn import mixture
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering 
from sklearn import tree
import graphviz
import random
from sklearn.semi_supervised import LabelSpreading
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from scipy import linalg
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import os
import pydotplus
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score

class Databases():

    def __init__(self):
        print("hello")
        # Set up basic df
        self.basic_df = pd.read_parquet('/Volumes/Britt_SSD/Location_Evals2_Beluga/comboEval.parquet')
        self.basic_df["isKeyOrContext"] = np.where(self.basic_df['keyNum']>0, 1, 0)
        
        # Set up basic numeric DF
        self.basic_num_df = self.basic_df
        self.basic_num_df = self.basic_num_df.applymap(lambda x: int(x) if isinstance(x, bool) else x)
        self.basic_num_df.drop(columns=["action_name","pacman_direction","red_ghost_direction","pink_ghost_direction","blue_ghost_direction","orange_ghost_direction","agent"], inplace=True)
        print(self.basic_num_df.dtypes[self.basic_num_df.dtypes != 'int64'][self.basic_num_df.dtypes != 'float64'])
        
        # Get list of five agent
        self.agent_names = self.basic_df.agent.unique()
        
        # Set up basic numeric DF
        self.basic_num_df.replace([np.inf, -np.inf], np.nan)
        self.basic_num_df.replace(np.nan, 0)
        df1 = self.basic_num_df[self.basic_num_df['agentNum']==1]
        df1.loc[:,"dbgNum"]=0
        df1.loc[:,"isDBG"]=0
        self.basic_num_df.update(df1)
        self.basic_num_df["isKeyOrContext"] = np.where(self.basic_num_df['keyNum']>0, 1, 0)
        
        # Set up super-slimmed DF
        self.df_num_slimmed = self.basic_num_df.drop(columns=['dark_blue_ghost1_coord_x', 'dark_blue_ghost1_coord_y', 'dark_blue_ghost2_coord_x', 'dark_blue_ghost2_coord_y', 'dark_blue_ghost3_coord_x', 'dark_blue_ghost3_coord_y','dark_blue_ghost4_coord_x', 'dark_blue_ghost4_coord_y', 'to_db1', 'to_db2', 'to_db3','to_db4', "context_state", "dbgNum",  "dark_blue_ghost4_y_relative", "dark_blue_ghost3_y_relative", "bigRewardNum", "blue_y_relative", "orange_y_relative", 'pacman_coord_x', 'pacman_coord_y', 'red_ghost_coord_x', 'red_ghost_coord_y', 'pink_ghost_coord_x', 'pink_ghost_coord_y', 'blue_ghost_coord_x', 'blue_ghost_coord_y', 'orange_ghost_coord_x', 'orange_ghost_coord_y', 'action 1 episode sum', 'action 1 total sum', 'action 2 episode sum', 'action 2 total sum', 'action 3 episode sum', 'action 3 total sum', 'action 4 episode sum', 'action 4 total sum', 'Unnamed: 0', 'key_state', 'keyNum', 'beforeLifeLoss', 'episode', 'epoch', 'epoch_step', 'end_of_episode',       'end_of_epoch', 'epoch_score', 'diff_to_dbg1','diff_to_dbg2', 'diff_to_dbg3', 'diff_to_dbg4', 'diff_to_pill1', 'diff_to_pill2', 'diff_to_pill3', 'diff_to_pill4', 'red_x_relative', 'red_y_relative', 'blue_x_relative', 'orange_x_relative', 'pink_x_relative', 'pink_y_relative', 'dark_blue_ghost1_x_relative', 'dark_blue_ghost1_y_relative', 'dark_blue_ghost2_x_relative', 'dark_blue_ghost2_y_relative',   'dark_blue_ghost3_x_relative', 'dark_blue_ghost4_x_relative'])

        # Set up super-full DF
        # add columns in dataframe
        self.df_num_full = self.basic_num_df
        self.df_num_full["reward_mean"] = self.basic_num_df.groupby('state').epoch_reward.transform('mean')
        self.df_num_full["reward_median"] = self.basic_num_df.groupby('state').epoch_reward.transform('median')
        self.df_num_full["reward_std"] = self.basic_num_df.groupby('state').epoch_reward.transform('std')
        self.df_num_full["episode_reward_mean"] = self.basic_num_df.groupby('episode').episode_reward.transform('mean')
        self.df_num_full["lives_mean"] = self.basic_num_df.groupby('epoch').lives.transform('mean')
        self.df_num_full["lives_median"] = self.basic_num_df.groupby('epoch').lives.transform('median')
        self.df_num_full["lives_std"] = self.basic_num_df.groupby('epoch').lives.transform('std')
        self.df_num_full["episode_actions_mean"] = self.basic_num_df.groupby('episode').action.transform('mean')
        self.df_num_full["episode_actions_median"] = self.basic_num_df.groupby('episode').action.transform('median')
        self.df_num_full["episode_actions_std"] = self.basic_num_df.groupby('episode').action.transform('std')
        self.df_num_full["epoch_actions_mean"] = self.basic_num_df.groupby('epoch').action.transform('std')
        self.df_num_full["epoch_actions_median"] = self.basic_num_df.groupby('epoch').action.transform('median')
        self.df_num_full["epoch_actions_std"] = self.basic_num_df.groupby('epoch').action.transform('std')

        # Df with just location information
        self.loc_DF = self.basic_num_df.drop(columns=['episode_reward', 'epoch_reward', 'total_reward','end_of_episode', 'end_of_epoch', 'episode', 'episode_step', 'epoch', 'epoch_step', 'state', 'mean_reward', 'action 1 episode sum', 'action 1 total sum', 'action 2 episode sum', 'action 2 total sum', 'action 3 episode sum', 'action 3 total sum', 'action 4 episode sum', 'action 4 total sum', 'epoch_score', 'Unnamed: 0', 'key_state', 'context_state', 'keyNum', 'bigRewardNum',  'dbgNum'])

        # Set up a df just for rewards and actions
        self.reward_df = self.basic_num_df.drop(columns=['to_pill_one', 'to_pill_two', 'to_pill_three', 'to_pill_four', 'to_red_ghost', 'to_pink_ghost', 'to_blue_ghost', 'to_orange_ghost', 'pacman_coord_x', 'pacman_coord_y', 'red_ghost_coord_x', 'red_ghost_coord_y', 'pink_ghost_coord_x', 'pink_ghost_coord_y', 'blue_ghost_coord_x', 'blue_ghost_coord_y', 'orange_ghost_coord_x', 'orange_ghost_coord_y', 'dark_blue_ghost1_coord_x', 'dark_blue_ghost1_coord_y', 'dark_blue_ghost2_coord_x', 'dark_blue_ghost2_coord_y','dark_blue_ghost3_coord_x', 'dark_blue_ghost3_coord_y', 'dark_blue_ghost4_coord_x', 'dark_blue_ghost4_coord_y', 'to_pill_mean', 'to_top_pills_mean', 'to_bottom_pills_mean', 'to_ghosts_mean', 'to_db1', 'to_db2', 'to_db3', 'to_db4', 'diff_to_red', 'diff_to_orange', 'diff_to_blue', 'diff_to_pink', 'diff_to_dbg1', 'diff_to_dbg2', 'diff_to_dbg3', 'diff_to_dbg4', 'diff_to_pill1', 'diff_to_pill2', 'diff_to_pill3', 'diff_to_pill4', 'red_x_relative', 'red_y_relative', 'blue_x_relative', 'blue_y_relative', 'orange_x_relative', 'orange_y_relative', 'pink_x_relative', 'pink_y_relative', 'dark_blue_ghost1_x_relative', 'dark_blue_ghost1_y_relative', 'dark_blue_ghost2_x_relative', 'dark_blue_ghost2_y_relative', 'dark_blue_ghost3_x_relative', 'dark_blue_ghost3_y_relative', 'dark_blue_ghost4_x_relative', 'dark_blue_ghost4_y_relative'])

       # And per-Agent DFs:
        self.agentNumeric_SlimmedDFs = []
        self.agentNumeric_FullDFs = []

        for index, num in enumerate(self.df_num_slimmed.agentNum.unique()):
            # Get just the normal DF with all info per agent
            tempNumDF = self.df_num_full[self.df_num_full['agentNum']==index]
            tempSlimmedDF = self.df_num_slimmed[self.df_num_slimmed['agentNum']==index]
            # Add to list of agent ground truths
            self.agentNumeric_FullDFs.append(tempNumDF)
            self.agentNumeric_SlimmedDFs.append(tempSlimmedDF)

        # Then over-sample each Agent's DF:
        oversample = SMOTE(sampling_strategy="minority")
        # Then get just the numeric DF per agent
        self.OS_agents = []
        # Combine all into test/train X and y
        self.OS_all = pd.DataFrame()

        # Set up SMOTE oversampled DFs for each agent
        for index, agentDF in enumerate(self.agentNumeric_SlimmedDFs):
            df = agentDF.drop(columns=["isKeyOrContext"])
            X_resampled, y_resampled = oversample.fit_resample(df, agentDF['isKeyOrContext'])
            
            X_resampled['isKeyOrContext'] = y_resampled
            self.OS_all = pd.concat([self.OS_all, X_resampled])
            
            self.OS_agents.append(X_resampled)
            
    def makeGaussianPlots(self, dfList, df_names_list, colorsList, folderName, algorithm_name, perAgent=False, interest=None):
        x_list = ['episode_reward', 'epoch_reward', 'total_reward',
       'episode_step', 'state', 'to_pill_three', 'to_pill_four', 'to_red_ghost',
       'to_pink_ghost', 'importance',
       'to_pill_mean', 'to_top_pills_mean', 'to_bottom_pills_mean',
       'to_ghosts_mean', 'agentNum']
        y_list = ['action', 'reward', 'lives', 'to_pill_one',
       'to_pill_two', 'to_blue_ghost', 'to_orange_ghost', 'importance',
       'to_pill_mean', 'to_top_pills_mean', 'to_bottom_pills_mean',
       'to_ghosts_mean', 'diff_to_red', 'diff_to_orange', 'diff_to_blue',
       'diff_to_pink', 'isDBG']
       
        if not os.path.exists(folderName):
            os.makedirs(folderName)

        for index, agentDF in enumerate(dfList):
            for x in x_list:
                for y in y_list:
                    if (x in agentDF):
                        if (y in agentDF):
                            plt.scatter(agentDF[x],agentDF[y], c=colorsList[index], alpha=0.8)
                            if ((interest != None) & (interest in agentDF)):
                                temp = agentDF[agentDF[interest] > 0]
                                plt.scatter(temp[x],temp[y], c='r')
                            plt.xlabel(x)
                            plt.ylabel(y)
                            if (perAgent == True):
                                plt.title(algorithm_name + " per Agent for " + db.agent_names[index])
                                filename = algorithm_name + db.agent_names[index] + x + y + ".png"
                            else:
                                plt.title(algorithm_name + df_names_list[index])
                                filename = df_names_list[index] + x + y + ".png"
                            plt.legend(loc="best")
                            filePath = os.path.join(folderName, filename)
                            plt.savefig(filePath)
                            plt.close()

    def makeTreeFiles(self, clfTree, folderName, filename, df, target, classNames):
        
        if not os.path.exists(folderName):
            os.makedirs(folderName)
            
        filepath = os.path.join(folderName, filename)
        
        dot_data = tree.export_graphviz(clfTree, out_file = None,
                              feature_names = df.loc[:, df.columns != target].columns,
                              class_names = classNames,
                              filled=True, rounded=True,
                              special_characters=True)
        # Draw graph
        graph = pydotplus.graph_from_dot_data(dot_data)
        
        graph.write_png(filepath)

if __name__ == '__main__':
    
    # Set up parser so we can make pictures
    parser = argparse.ArgumentParser()
    #Britt additions
    parser.add_argument('--corr', action='store_true', help='run the algorithm to convert a tensorflow model to keras model')
    parser.set_defaults(convert_model=False)
    parser.add_argument('--randomtrees', action='store_true', help='output a stream into a new folder')
    parser.set_defaults(generate_stream=False)
    parser.add_argument('--trees', action='store_true', help='use to overlay a saliency map onto the screenshots')
    parser.set_defaults(overlay_saliency=False)
    parser.add_argument('--kmeans', action='store_true', help='run the algorithm to convert a tensorflow model to keras model')
    parser.set_defaults(convert_model=False)
    parser.add_argument('--clustering', action='store_true', help='output a stream into a new folder')
    parser.set_defaults(generate_stream=False)
    parser.add_argument('--gaussian', action='store_true', help='use to overlay a saliency map onto the screenshots')
    parser.set_defaults(overlay_saliency=False)
    parser.add_argument('--graphs', action='store_true', help='use to overlay a saliency map onto the screenshots')
    parser.set_defaults(overlay_saliency=False)
    parser.add_argument('--random', action='store_true', help='use to overlay a saliency map onto the screenshots')
    parser.set_defaults(overlay_saliency=False)
    parser.add_argument('--isolation', action='store_true', help='use to overlay a saliency map onto the screenshots')
    parser.set_defaults(overlay_saliency=False)
    parser.add_argument('--vis', action='store_true', help='use to overlay a saliency map onto the screenshots')
    parser.set_defaults(overlay_saliency=False)
    
    args = parser.parse_args()

    
############################################################################################################
    # Make a databases object
    db = Databases()
    
############################################################################################################
    if args.corr:
    
        # Correlation Matrix of Entire Oversampled DB
        corrMatrix = db.OS_all.corr()
        df = db.OS_all
        f = plt.figure(figsize=(19, 15))
        plt.matshow(corrMatrix, fignum=f.number)
        plt.xticks(range(len(df.columns)), df.columns, fontsize=11, rotation=45)
        plt.yticks(range(len(df.columns)), df.columns, fontsize=11)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix of Initial DF', fontsize=16);
        plt.savefig("HeatmapOfAllOversampled.png")
        plt.close()

        # Correlation Matrix of Slimmed Numeric DF
        corrMatrix = db.df_num_slimmed.corr()
        df = db.df_num_slimmed
        f = plt.figure(figsize=(19, 15))
        plt.matshow(corrMatrix, fignum=f.number)
        plt.xticks(range(len(df.columns)), df.columns, fontsize=11, rotation=45)
        plt.yticks(range(len(df.columns)), df.columns, fontsize=11)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix of Slimmed DF', fontsize=16);
        plt.savefig("HeatmapOfSlimmedDatabase.png")
        plt.close()

        for index, agent in enumerate(db.agentNumeric_FullDFs):
            # Correlation Matrix of Full Numeric DF
            corrMatrix = agent.corr()
            df = agent
            f = plt.figure(figsize=(19, 15))
            plt.matshow(corrMatrix, fignum=f.number)
            plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
            plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=14)
            plt.title('Correlation Matrix of Slimmed DF' + str(db.agent_names[index]), fontsize=16);
            filename = "HeatmapOfFullDatabase" + str(db.agent_names[index]) + ".png"
            plt.savefig(filename)
            plt.close()

        for index, agent in enumerate(db.agentNumeric_SlimmedDFs):
            # Correlation Matrix of Full Numeric DF
            corrMatrix = agent.corr()
            df = agent
            f = plt.figure(figsize=(19, 15))
            plt.matshow(corrMatrix, fignum=f.number)
            plt.xticks(range(len(df.columns)), df.columns, fontsize=11, rotation=45)
            plt.yticks(range(len(df.columns)), df.columns, fontsize=11)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=14)
            plt.title('Correlation Matrix of Slimmed DF ' + str(db.agent_names[index]), fontsize=16);
            filename = "HeatmapOfSlimmedDatabase" + str(db.agent_names[index]) + ".png"
            plt.savefig(filename)
            plt.close()

        for index, agent in enumerate(db.OS_agents):
            # Correlation Matrix of Full Numeric DF
            corrMatrix = agent.corr()
            df = agent
            f = plt.figure(figsize=(19, 15))
            plt.matshow(corrMatrix, fignum=f.number)
            plt.xticks(range(len(df.columns)), df.columns, fontsize=11, rotation=45)
            plt.yticks(range(len(df.columns)), df.columns, fontsize=11)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=14)
            plt.title('Correlation Matrix of Slimmed DF ' + str(db.agent_names[index]), fontsize=16);
            filename = "HeatmapOfOverSampledDatabase" + str(db.agent_names[index]) + ".png"
            plt.savefig(filename)
            plt.close()
            
        for index, agent in enumerate(db.OS_agents):
            # Ghost Correlation
            df = agent
            f = plt.figure(figsize=(19, 15))
#            plt.scatter("state", "to_pink_ghost", data=df, label = "pink",alpha=0.3, c="pink")
#            plt.scatter("state", "to_orange_ghost", data=df, label = "pink",alpha=0.3, c="orange")
#            plt.scatter("state", "to_blue_ghost", data=df, label = "pink",alpha=0.3, c="blue")
#            plt.scatter("state", "to_red_ghost", data=df, alpha=0.3, c="r", label="red")
            plt.scatter("state", "diff_to_pink", data=df, label = "pink",alpha=0.6, c="pink")
            plt.scatter("state", "diff_to_orange", data=df, label = "orange",alpha=0.5, c="orange")
            plt.scatter("state", "diff_to_blue", data=df, label = "blue",alpha=0.4, c="blue")
            plt.scatter("state", "diff_to_red", data=df, alpha=0.1, c="r", label="red")
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=14)
            plt.title('Ghost Distance Relationships ' + str(db.agent_names[index]), fontsize=16);
            filename = "GhostDistances" + str(db.agent_names[index]) + ".png"
            plt.savefig(filename)
            plt.close()
    
############################################################################################################
    if args.kmeans:
    
        fullDF_List = [db.OS_all, db.df_num_slimmed]
        df_names = ["Oversampled", "Slimmed"]
        targets = ['isKeyOrContext', 'agentNum']
        target_names = ["Key States", "Agent Number"]
        
        for df_index, df in enumerate(fullDF_List):
            for target_index, target in enumerate(targets):
                # try kmeans on entire (slimmed, for computational ease) DF
                response = df[target]
                temp = df.drop(columns=[target])
                
                X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
                
                if (target_index == 0):
                    kmeans = MiniBatchKMeans(n_clusters=2, batch_size=31)
                else:
                    kmeans = MiniBatchKMeans(n_clusters=5, batch_size=1000)
                
                clus = kmeans.fit(X_train)
                result = kmeans.predict(X_test)
                
                print("KMeans on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                print(np.sum(y_test-result))
                cf = confusion_matrix(y_test, result)
                
                print("Confusion Matrix: ")
                print(cf)
                print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
                
                plt.scatter(X_test["state"],y_test, c=result)
                plt.xlabel("time step")
                plt.ylabel(target_names[target_index])
                plt.title("KMeans on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                filename = "KMeans" + target_names[target_index] + "_" + df_names[df_index] + ".png"
                plt.savefig(filename)
                plt.clf()
                plt.close()
            
        kmeans = MiniBatchKMeans(n_clusters=2, batch_size=31)
        
#        # try kmeans on Normal Per-agent
#        for index, agentDF in enumerate(db.agentNumeric_FullDFs):
#            response = agentDF['isKeyOrContext']
#            temp = agentDF.drop(columns=['isKeyOrContext'])
#            X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
#
#            clus = kmeans.fit(X_train)
#            result = kmeans.predict(X_test)
#            print("Running Full DF K-Means for Agent: ")
#            print(db.agent_names[index])
#            print(np.sum(y_test-result))
#            cf = confusion_matrix(y_test, result)
#            print("Confusion Matrix: ")
#            print(cf)
#            print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
#            plt.scatter(X_test["state"],y_test, c=result)
#            plt.xlabel("time step")
#            plt.ylabel("key state")
#            plt.title("Full Info K-Means: " + db.agent_names[index])
#            filename = "FullDFsKmeans" + str(db.agent_names[index]) + ".png"
#            plt.savefig(filename)
#            plt.clf()
#            plt.close()
#
#
#        #Slimmed per agent
#        for index, agentDF in enumerate(db.agentNumeric_SlimmedDFs):
#            response = agentDF['isKeyOrContext']
#            temp = agentDF.drop(columns=['isKeyOrContext'])
#            X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
#
#            clus = kmeans.fit(X_train)
#            result = kmeans.predict(X_test)
#            print("Running Slimmed DF K-Means for Agent: ")
#            print(db.agent_names[index])
#            print(np.sum(y_test-result))
#            cf = confusion_matrix(y_test, result)
#            print("Confusion Matrix: ")
#            print(cf)
#            print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
#            plt.scatter(X_test["state"],y_test, c=result)
#            plt.xlabel("time step")
#            plt.ylabel("key state")
#            plt.title("Selected Features K-Means: " + db.agent_names[index])
#            filename = "SlimmedDFsKmeans" + str(db.agent_names[index]) + ".png"
#            plt.savefig(filename)
#            plt.clf()
#            plt.close()
#
#        #Over-Sampled Slimmed per agent
#        for index, agentDF in enumerate(db.agentNumeric_SlimmedDFs):
#            response = agentDF['isKeyOrContext']
#            temp = agentDF.drop(columns=['isKeyOrContext'])
#            X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
#
#            clus = kmeans.fit(X_train)
#            result = kmeans.predict(X_test)
#            print("Running Over-Sampled DF K-Means for Agent: ")
#            print(db.agent_names[index])
#            print(np.sum(y_test-result))
#            # order: confusion_matrix(y_true, y_pred)
#            cf = confusion_matrix(y_test, result)
#            print("Confusion Matrix: ")
#            print(cf)
#            print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
#            plt.scatter(X_test["state"],y_test, c=result)
#            plt.xlabel("time step")
#            plt.ylabel("key state")
#            plt.title("Over-Sampled K-Means: " + db.agent_names[index])
#            filename = "OverSampledDFsKmeans" + str(db.agent_names[index]) + ".png"
#            plt.savefig(filename)
#            plt.clf()
#            plt.close()
            
############################################################################################################
    if args.clustering:
        fullDF_List = [db.OS_all, db.df_num_slimmed]
        df_names = ["Oversampled", "Slimmed"]
        targets = ['isKeyOrContext', 'agentNum']
        target_names = ["Key States", "Agent Number"]
        
        #Mean Shift
        clustering = MeanShift(min_bin_freq=10, max_iter=400)
        
        for df_index, df in enumerate(fullDF_List):
            for target_index, target in enumerate(targets):
                # try kmeans on entire (slimmed, for computational ease) DF
                response = df[target]
                temp = df.drop(columns=[target])
                X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
                
                clus = clustering.fit(X_train)
                result = clustering.predict(X_test)
                
                print("Mean Shift on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                print(np.sum(y_test-result))
                # order: confusion_matrix(y_true, y_pred)
                cf = confusion_matrix(y_test, result)
                
                print("Confusion Matrix: ")
                print(cf)
                print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
                
                plt.scatter(X_test["state"],y_test, c=result)
                plt.xlabel("time step")
                plt.ylabel(target_names[target_index])
                plt.title("Mean Shift on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                
                filename = "MeanShift" + target_names[target_index] + "_" + df_names[df_index] + ".png"
                plt.savefig(filename)
                plt.clf()
                plt.close()
            
        # DBSCAN on Slimmed per agent
        clustering = DBSCAN(eps=10, min_samples=5000)
        for index, agentDF in enumerate(db.agentNumeric_SlimmedDFs):
            clus = clustering.fit(agentDF)
            result = clus.labels_
            print("Using DBSCAN Clustering on Agent: " + str(db.agent_names[index]))
            print(result)
            plt.scatter(agentDF["state"],agentDF['agentNum'], c=result)
            plt.xlabel("time step")
            plt.ylabel("agent number")
            filename = "DBSCANSlimmed" + str(db.agent_names[index]) + ".png"
            plt.savefig(filename)
            plt.clf()
            plt.close()
            
#        fullDF_List = [db.OS_all, db.df_num_slimmed]
#        targets = ['isKeyOrContext', 'agentNum']

        # OPTICS on Slimmed All Together
        for df_index, df in enumerate(fullDF_List):
            for target_index, target in enumerate(targets):
                if (target_index == 0):
                    clustering = OPTICS(min_samples=15, min_cluster_size=31)
                else:
                    clustering = OPTICS(min_samples=5000, min_cluster_size=9000)
                
                response = df[target]
                temp = df.drop(columns=[target])
                X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
                
                clus = clustering.fit(X_train)
                result = clustering.predict(X_test)
                                
                print("OPTICS on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                print(np.sum(y_test-result))
                # order: confusion_matrix(y_true, y_pred)
                cf = confusion_matrix(y_test, result)
                
                print("Confusion Matrix: ")
                print(cf)
                print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
                
                plt.scatter(X_test["state"],y_test, c=result)
                plt.xlabel("time step")
                plt.ylabel(target_names[target_index])
                plt.title("OPTICS on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                
                filename = "OPTICS" + target_names[target_index] + "_" + df_names[df_index] + ".png"
                plt.savefig(filename)
                plt.clf()
                plt.close()

############################################################################################################
    if args.gaussian:
        fullDF_List = [db.OS_all, db.df_num_slimmed]
        df_names = ["Oversampled", "Slimmed"]
        targets = ['isKeyOrContext', 'agentNum']
        target_names = ["Key States", "Agent Number"]

        clustering = mixture.GaussianMixture(n_components=10, covariance_type='tied', max_iter=20, n_init=10)
        
        # Gaussian Clustering
        for df_index, df in enumerate(fullDF_List):
            for target_index, target in enumerate(targets):
                gaussian_results = []
                testDF_List = []
                
                response = df[target]
                temp = df.drop(columns=[target])
                X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
                
                clus = clustering.fit(X_train)
                result = clustering.predict(X_test)
                gaussian_results.append(result)
                testDF_List.append(X_test)
                
                print("Gaussian on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                print(np.sum(y_test-result))
                # order: confusion_matrix(y_true, y_pred)
                cf = confusion_matrix(y_test, result)
                
                print("Confusion Matrix: ")
                print(cf)
                print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
                
                plt.scatter(X_test["state"],y_test, c=result)
                plt.xlabel("Time Step")
                plt.ylabel(target_names[target_index])
                plt.title("Gaussian on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                
                filename = "Gaussian" + target_names[target_index] + "_" + df_names[df_index] + ".png"
                plt.savefig(filename)
                plt.clf()
                plt.close()
        # Need to call Gaussian Plotting Here
        db.makeGaussianPlots(testDF_List, df_names, gaussian_results, "/Users/byrdsmyth/Documents/School/Classes/CPTS570/Project/codebase/AllAgentsGaussian", "Gaussian", perAgent=False, interest=None)
        
        agentDFsList = [db.agentNumeric_FullDFs, db.OS_agents, db.agentNumeric_SlimmedDFs]
        df_names = ["Full","Oversampled","Slimmed"]
        
        for list_index, dfList in enumerate(agentDFsList):
            gaussian_results = []
            testDF_List = []
            for agent_index, agentDF in enumerate(dfList):
                response = df['isKeyOrContext']
                temp = df.drop(columns=['isKeyOrContext'])
                X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
                
                clus = clustering.fit(X_train)
                result = clustering.predict(X_test)
                gaussian_results.append(result)
                testDF_List.append(X_test)
                
                print("Gaussian on Key States for Agent " + db.agent_names[agent_index] + " using " + df_names[list_index] + " DF")
                print(np.sum(y_test-result))
                # order: confusion_matrix(y_true, y_pred)
                cf = confusion_matrix(y_test, result)
                print("Confusion Matrix: ")
                print(cf)
                print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
                
                plt.scatter(X_test["state"],y_test, c=result)
                plt.xlabel("Time Step")
                plt.ylabel("Key States")
                plt.title("Gaussian on Key States for Agent " + db.agent_names[agent_index] + " using " + df_names[list_index] + " DF")
                
                filename = "Gaussian" + db.agent_names[agent_index] + "_" + df_names[list_index] + ".png"
                plt.savefig(filename)
                plt.clf()
                plt.close()
            # Need to call Gaussian Plotting Here
            db.makeGaussianPlots(testDF_List, [], gaussian_results, "/Users/byrdsmyth/Documents/School/Classes/CPTS570/Project/codebase/PerAgentGaussian", "Gaussian", perAgent=True, interest=None)

############################################################################################################
    # Semi-Supervised
    # LabelSpreading minimizes a loss function that has
    # regularization properties, as such it is often more robust to noise.


    # Just note this broke everything so we ended up not able to do it

############################################################################################################
    if args.randomtrees:
        fullDF_List = [db.OS_all, db.df_num_slimmed]
        df_names = ["Oversampled", "Slimmed"]
        targets = ['isKeyOrContext', 'agentNum', 'action']
        target_names = ["Key States", "Agent Number", "Action Taken"]
        
        clustering1 = RandomForestClassifier(max_depth=5)
        clustering2 = RandomForestRegressor(max_depth=5)
        
        clustering_algos = [clustering1, clustering2]
        
        clus_names = ["RandomForestClassifier", "RandomForestRegressor"]
        
        for clus_index, clustering in enumerate(clustering_algos):
            # Gaussian Clustering
            for df_index, df in enumerate(fullDF_List):
                for target_index, target in enumerate(targets):
                    isolationForest_results = []
                    xTestList = []
                    
                    response = df[target]
                    response = response.astype(int)
                    temp = df.drop(columns=[target])
                    X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
                    
                    clus = clustering.fit(X_train, y_train)
                    result = clustering.predict(X_test)
                    isolationForest_results.append(result)
                    xTestList.append(X_test)
                                    
                    print(clus_names[clus_index] + " on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                    print(np.sum(y_test-result))
                    # order: confusion_matrix(y_true, y_pred)
                    cf = confusion_matrix(y_test, result)
                    
                    if (target_index == 0):
                        print("FPR = 1 - TNR and TNR = specificity")
                        print("FNR = 1 - TPR and TPR = recall")
                        print("average precision recall : high precision relates to a low false positive rate, and high recall relates to a low false negative rate")
                        print(average_precision_score(y_test, result))
                        print("precision is the ratio tp / (tp + fp)")
                        print("the ability of the classifier not to label as positive a sample that is negative")
                        print("precision: ")
                        print(precision_score(y_test, result))
                    
                    print("Confusion Matrix: ")
                    print(cf)
                    print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
                    
                    plt.scatter(X_test["state"],y_test, c=result)
                    plt.xlabel("time step")
                    plt.ylabel(target_names[target_index])
                    plt.title(clus_names[clus_index] + " on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                    
                    filename = clus_names[clus_index] + target_names[target_index] + "_" + df_names[df_index] + ".png"
                    plt.savefig(filename)
                    plt.clf()
                    plt.close()
            # Need to call Gaussian Plotting Here
            #db.makeGaussianPlots(xTestList, df_names, isolationForest_results, "/Users/byrdsmyth/Documents/School/Classes/CPTS570/Project/codebase/AllAgents" + clus_names[clus_index], clus_names[clus_index], perAgent=False, interest=None)
            
            agentDFsList = [db.agentNumeric_FullDFs, db.OS_agents, db.agentNumeric_SlimmedDFs]
            df_names = ["Full","Oversampled","Slimmed"]
            
            targets = ['isKeyOrContext', 'action']
            target_names = ["Key States", "Action Taken"]
            
            for list_index, dfList in enumerate(agentDFsList):
                isolationForest_results = []
                for agent_index, agentDF in enumerate(dfList):
                    for target_index, target in enumerate(targets):
                        response = agentDF[target]
                        response = response.astype(int)
                        temp = agentDF.drop(columns=[target])
                        X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
                        
                        clus = clustering.fit(X_train, y_train)
                        result = clustering.predict(X_test)
                        isolationForest_results.append(result)
                                        
                        print(clus_names[clus_index] + " on " + target_names[target_index] + " for Agent " + db.agent_names[agent_index] + " using " + df_names[list_index] + " DF")
                        print(np.sum(y_test-result))
                        # order: confusion_matrix(y_true, y_pred)
                        cf = confusion_matrix(y_test, result)
                        print("Confusion Matrix: ")
                        print(cf)
                        print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
                        
                        if (target_index == 0):
                            print("FPR = 1 - TNR and TNR = specificity")
                            print("FNR = 1 - TPR and TPR = recall")
                            print("average precision recall : high precision relates to a low false positive rate, and high recall relates to a low false negative rate")
                            print(average_precision_score(y_test, result))
                            print("precision is the ratio tp / (tp + fp)")
                            print("the ability of the classifier not to label as positive a sample that is negative")
                            print("precision: ")
                            print(precision_score(y_test, result))
                            
                        plt.scatter(X_test["state"],y_test, c=result)
                        plt.xlabel("Time Step")
                        plt.ylabel("Key States")
                        plt.title(clus_names[clus_index] + " on " + target_names[target_index] + " for Agent " + db.agent_names[agent_index] + " using " + df_names[list_index] + " DF")
                        
                        filename = clus_names[clus_index] + db.agent_names[agent_index] + "_" + df_names[list_index] + "_" + target_names[target_index] + ".png"
                        plt.savefig(filename)
                        plt.clf()
                        plt.close()
                    # Need to call Gaussian Plotting Here
                    #db.makeGaussianPlots(dfList, [], isolationForest_results, "/Users/byrdsmyth/Documents/School/Classes/CPTS570/Project/codebase/PerAgent" + clus_names[clus_index] + target_names[target_index], clus_names[clus_index], perAgent=False, interest=None)
        

############################################################################################################
#        fullDF_List = [db.OS_all, db.df_num_slimmed]
#        df_names = ["Oversampled", "Slimmed"]
#        targets = ['isKeyOrContext', 'agentNum']
#        target_names = ["Key States", "Agent Number"]
    if args.trees:
        fullDF_List = [db.OS_all, db.df_num_slimmed]
        df_names = ["Oversampled", "Slimmed"]
        targets = ['isKeyOrContext', 'agentNum', 'action']
        target_names = ["Key States", "Agent Number", "Action Taken"]
        
        clustering = tree.DecisionTreeClassifier(max_depth=15)
        
        action_class_names = ['UP', 'RIGHT', 'LEFT', 'DOWN']

        # Normal Trees
        for df_index, df in enumerate(fullDF_List):
            for target_index, target in enumerate(targets):
                key_class_names = ["isNotKey", "isKey"]
                print(key_class_names)
                agent_class_names = db.agent_names
                
                class_names_list = [key_class_names, agent_class_names, action_class_names]
            
                response = df[target]
                response = response.astype(int)
                temp = df.drop(columns=[target])
                if (target_index == 0):
                    if ('keyNum' in temp):
                        temp.drop(columns=['keyNum'], inplace = True)
                    if ('key_state' in temp):
                        temp.drop(columns=['key_state'], inplace = True)
                    if ('context_state' in temp):
                        temp.drop(columns=['context_state'], inplace = True)
            
                X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
                
                clus = clustering.fit(X_train, y_train)
                result = clustering.predict(X_test)
                                
                print("Decision Tree on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                print(np.sum(y_test-result))
                # order: confusion_matrix(y_true, y_pred)
                cf = confusion_matrix(y_test, result)
                
                print("Confusion Matrix: ")
                print(cf)
                print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
                if (target_index == 0):
                    print("FPR = 1 - TNR and TNR = specificity")
                    print("FNR = 1 - TPR and TPR = recall")
                    print("average precision recall : high precision relates to a low false positive rate, and high recall relates to a low false negative rate")
                    print(average_precision_score(y_test, result))
                    print("precision is the ratio tp / (tp + fp)")
                    print("the ability of the classifier not to label as positive a sample that is negative")
                    print("precision: ")
                    print(precision_score(y_test, result))
                
                plt.scatter(X_test["state"],y_test, c=result)
                plt.xlabel("time step")
                plt.ylabel(target_names[target_index])
                plt.title("Decision Tree on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                
                filename = "DecisionTree" + target_names[target_index] + "_" + df_names[df_index] + ".png"
                plt.savefig(filename)
                plt.clf()
                plt.close()
                filename = str(df_names[df_index]) + "_" + str(target_names[target_index]) + "TREE15.png"
                # Need to call Tree Graphing Plot
                db.makeTreeFiles(clustering, folderName = "/Users/byrdsmyth/Documents/School/Classes/CPTS570/Project/codebase/AllAgentsTrees", filename = filename, df = temp, target = target, classNames = class_names_list[target_index])
        
        agentDFsList = [db.agentNumeric_FullDFs, db.OS_agents, db.agentNumeric_SlimmedDFs]
        df_names = ["Full","Oversampled","Slimmed"]
        
        targets = ['isKeyOrContext', 'action']
        target_names = ["Key States", "Action Taken"]
        class_names_list = [key_class_names, action_class_names]
        
        for list_index, dfList in enumerate(agentDFsList):
            for agent_index, agentDF in enumerate(dfList):
                for target_index, target in enumerate(targets):
                    response = agentDF[target]
                    response = response.astype(int)
                    temp = agentDF.drop(columns=[target])
                    if (target_index == 0):
                        if ('keyNum' in temp):
                            temp.drop(columns=['keyNum'], inplace = True)
                        if ('key_state' in temp):
                            temp.drop(columns=['key_state'], inplace = True)
                        if ('context_state' in temp):
                            temp.drop(columns=['context_state'], inplace = True)
                    X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)
                    
                    clus = clustering.fit(X_train, y_train)
                    result = clustering.predict(X_test)
                                    
                    print("Decision Tree on " + target_names[target_index] + " for Agent " + db.agent_names[agent_index] + " using " + df_names[list_index] + " DF")
                    print(np.sum(y_test-result))
                    # order: confusion_matrix(y_true, y_pred)
                    cf = confusion_matrix(y_test, result)
                    print("Confusion Matrix: ")
                    print(cf)
                    print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")
                    if (target_index == 0):
                        print("FPR = 1 - TNR and TNR = specificity")
                        print("FNR = 1 - TPR and TPR = recall")
                        print("average precision recall : high precision relates to a low false positive rate, and high recall relates to a low false negative rate")
                        print(average_precision_score(y_test, result))
                        print("precision is the ratio tp / (tp + fp)")
                        print("the ability of the classifier not to label as positive a sample that is negative")
                        print("precision: ")
                        print(precision_score(y_test, result))
                    
                    plt.scatter(X_test["state"],y_test, c=result)
                    plt.xlabel("Time Step")
                    plt.ylabel("Key States")
                    plt.title("Decision Tree on " + target_names[target_index] + " for Agent " + db.agent_names[agent_index] + " using " + df_names[list_index] + " DF")
                    
                    filename = "DecisionTree" + db.agent_names[agent_index] + "_" + df_names[list_index] + ".png"
                    plt.savefig(filename)
                    plt.clf()
                    plt.close()
                    
                    # Need to call Tree Graphing Plot
                    db.makeTreeFiles(clustering, folderName = "/Users/byrdsmyth/Documents/School/Classes/CPTS570/Project/codebase/PerAgentTrees", filename = db.agent_names[agent_index] + "_" + df_names[list_index] + target_names[target_index] + "TREE15.png", df = temp, target = target, classNames = class_names_list[target_index])

############################################################################################################
#        fullDF_List = [db.OS_all, db.df_num_slimmed]
#        df_names = ["Oversampled", "Slimmed"]
#        targets = ['isKeyOrContext', 'agentNum']
#        target_names = ["Key States", "Agent Number"]
#    if args.random:
#        # Try Random Pulls From Data
#        # Lets see about comparing the key states to randomly drawn ranges
#        tempList = []
#        tpDF = df_num[df_num["isKeyOrContext"]==0]
#        for i in range(0,100):
#            for index, agent in enumerate(tpDF.agentNum.unique()):
#                tmp = tpDF[tpDF['agentNum']==agent]
#                print(tmp.agentNum.unique())
#                for i in range(0,1):
#                    tempIndex = random.randint(0, len(df))
#                    print(tempIndex)
#
#                    if tempIndex < 31:
#                        temp = tpDF[tempIndex:tempIndex+31]
#                    else:
#                        temp = tpDF[tempIndex-31:tempIndex]
#                    tempList.append(temp)

        # Visualize Outcome
        

############################################################################################################
    if args.graphs:
        fullDF_List = [db.OS_all, db.df_num_slimmed]
        df_names = ["Oversampled", "Slimmed"]
        targets = ['isKeyOrContext', 'agentNum']
        target_names = ["Key States", "Agent Number"]
        
        # Graphs
        for index, num in enumerate(df.agentNum.unique()):
            G = nx.from_pandas_edgelist(agentNumericDFs[index], source='episode', target='keyNum')
            leaderboard = {}
            for x in G.nodes:
                leaderboard[x] = len(G[x])
            s = pd.Series(leaderboard, name='connections')
            df2 = s.to_frame().sort_values('connections', ascending=False)
            print(df2)
            
            plt.title(db.agent_names[index])
            nx.draw_shell(G, with_labels=True)
            plt.show()

        # http://jonathansoma.com/lede/algorithms-2017/classes/networks/networkx-graphs-from-source-target-dataframe/
        # 1. Create the graph
        plt.figure(figsize=(12, 12))
        g = nx.from_pandas_edgelist(df, source='reward', target='keyNum')

        # 2. Create a layout for our nodes
        layout = nx.spring_layout(g,iterations=50)

        # 3. Draw the parts we want
        nx.draw_networkx_edges(g, layout, edge_color='#AAAAAA')

        clubs = [node for node in g.nodes() if node in df.agentNum.unique() and g.degree(node) > 1]
        size = [g.degree(node) * 80 for node in g.nodes() if node in df.agentNum.unique()]
        nx.draw_networkx_nodes(g, layout, nodelist=clubs, node_size=size, node_color='lightblue')

        people = [node for node in g.nodes() if node in df.reward.unique() and g.degree(node) > 5]
        nx.draw_networkx_nodes(g, layout, nodelist=people, node_size=100, node_color='#AAAAAA')

        high_degree_people = [node for node in g.nodes() if node in df.reward.unique() and g.degree(node) > 10]
        nx.draw_networkx_nodes(g, layout, nodelist=high_degree_people, node_size=100, node_color='#fc8d62')

        club_dict = dict(zip(clubs, clubs))
        nx.draw_networkx_labels(g, layout, labels=club_dict)

        # 4. Turn off the axis because I know you don't want it
        plt.axis('off')

        plt.title("Revolutionary Clubs")

        # 5. Tell matplotlib to show it
        plt.show()


        # In[ ]:


        # Weighted Graph
        for index, num in enumerate(df_num_slimmed.agentNum.unique()):
            G = nx.from_pandas_edgelist(train_temp, source='lives', target=train_response, edge_attr='to_ghosts_mean')
        #     G = nx.Graph(X_red)

            print(G.edges())

            elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["to_ghosts_mean"] > 100.75]
            esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["to_ghosts_mean"] <= 100.75]

            pos = nx.spring_layout(G)  # positions for all nodes

            # nodes
            nx.draw_networkx_nodes(G, pos, node_size=700)

            # edges
            nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
            nx.draw_networkx_edges(
                G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
            )

            # labels
            nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

            plt.axis("off")
            plt.show()



############################################################################################################
# Anomoly Detection - based on guest speaker's presentation
    if args.isolation:
        fullDF_List = [db.OS_all, db.df_num_slimmed]
        df_names = ["Oversampled", "Slimmed"]
        targets = ['isKeyOrContext', 'agentNum']
        target_names = ["Key States", "Agent Number"]

        clustering = IsolationForest(contamination = 0.05)
        
        # Isolation Forests
        for df_index, df in enumerate(fullDF_List):
            for target_index, target in enumerate(targets):
                isolationForest_results = []
                testDF_List = []

                response = df[target]
                temp = df.drop(columns=[target])
                X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)

                clus = clustering.fit(X_train)
                result = clustering.predict(X_test)
                isolationForest_results.append(result)
                testDF_List.append(X_test)

                print("Isolation Forest on " + target_names[target_index] + " using " + df_names[df_index] + " DF")
                print(np.sum(y_test-result))
                # order: confusion_matrix(y_true, y_pred)
                cf = confusion_matrix(y_test, result)

                print("Confusion Matrix: ")
                print(cf)
                print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")

                plt.scatter(X_test["state"],X_test["total_reward"], c=result)
                if ('isKeyOrContext' in X_test):
                    temp = X_test[X_test['isKeyOrContext'] > 0]
                    plt.scatter(temp["state"],temp['total_reward'], c="r")

                plt.xlabel("time step")
                plt.ylabel(target_names[target_index])
                plt.title("Isolation Forest on " + target_names[target_index] + " using " + df_names[df_index] + " DF")

                filename = "Isolation Forest" + target_names[target_index] + "_" + df_names[df_index] + ".png"
                plt.savefig(filename)
                plt.clf()
                plt.close()
        # Need to call Gaussian Plotting Here
        db.makeGaussianPlots(testDF_List, df_names, isolationForest_results, "/Users/byrdsmyth/Documents/School/Classes/CPTS570/Project/codebase/AllAgentsIsolationForest", perAgent=False, interest='isKeyOrContext')
        
        agentDFsList = [db.agentNumeric_FullDFs, db.OS_agents, db.agentNumeric_SlimmedDFs]
        df_names = ["Full","Oversampled","Slimmed"]

        for list_index, dfList in enumerate(agentDFsList):
            isolationForest_results = []
            testDF_List = []
            for agent_index, agentDF in enumerate(dfList):
                if (agent_index > 3):
                    response = agentDF['isKeyOrContext']
                    temp = agentDF.drop(columns=['isKeyOrContext'])
                    X_train, X_test, y_train, y_test = train_test_split(temp, response, test_size=0.2)

                    clus = clustering.fit(X_train)
                    result = clustering.predict(X_test)
                    isolationForest_results.append(result)
                    X_test['isKeyOrContext'] = y_test
                    testDF_List.append(X_test)

                    print("Isolation Forest on Key States for Agent " + db.agent_names[agent_index] + " using " + df_names[list_index] + " DF")
                    print(np.sum(y_test-result))
                    # order: confusion_matrix(y_true, y_pred)
                    cf = confusion_matrix(y_test, result)
                    print("Confusion Matrix: ")
                    print(cf)
                    print("i-th row & j-th column indicates the number of samples with true label being i-th class and prediced label being j-th class.")

                    plt.scatter(X_test["state"],y_test, c=result)
                    plt.xlabel("Time Step")
                    plt.ylabel("Key States")
                    plt.title("Gaussian on Key States for Agent " + db.agent_names[agent_index] + " using " + df_names[list_index] + " DF")

                    filename = "Isolation Forest" + db.agent_names[agent_index] + "_" + df_names[list_index] + ".png"
                    plt.savefig(filename)
                    plt.clf()
                    plt.close()
                # Need to call Gaussian Plotting Here
                db.makeGaussianPlots(testDF_List, [], isolationForest_results, "/Users/byrdsmyth/Documents/School/Classes/CPTS570/Project/codebase/PerAgentIsolationForest", "Isolation Forest", perAgent=True, interest='isKeyOrContext')



        temp = db.OS_all
        temp['keyNum'] = db.basic_num_df['keyNum']
        fullDF_List = [temp, db.df_num_full]
        df_names = ["Oversampled", "Full"]
        targets = ['isKeyOrContext', 'agentNum']
        target_names = ["Key States", "Agent Number"]

        clustering = IsolationForest(contamination = 0.05)
        
        # Gaussian Clustering
        for df_index, df in enumerate(fullDF_List):
            for target_index, target in enumerate(targets):
                for index, num in enumerate(df.agentNum.unique()):
                    tempTrain = df[df['agentNum']==index]
                    clf = IsolationForest(contamination=0.03).fit(tempTrain)
                    outliers = clf.predict(tempTrain)
                    tempTrain["out"] = outliers
                    plt.scatter("state", "total_reward", c="cyan", alpha=0.005, data = tempTrain[tempTrain["out"]==1])
                    plt.scatter("state", "total_reward", alpha=0.45, c="g", data = tempTrain[tempTrain["out"]==-1])
                    plt.scatter("state", "total_reward", c="r", alpha=0.75, data = tempTrain[tempTrain['keyNum']>0])
                    plt.title(db.agent_names[index] + " " + df_names[df_index] + " " + target_names[target_index])
                    filename = "IsolationForestOutlierDetectionWithAllAgentDF" + db.agent_names[index] + df_names[df_index] + target_names[target_index] + ".png"
                    plt.savefig(filename)
                    plt.clf()
                    plt.close()
        
        agentDFsList = [db.agentNumeric_FullDFs, db.OS_agents, db.agentNumeric_SlimmedDFs]
        df_names = ["Full","Oversampled","Slimmed"]
        
        for list_index, dfList in enumerate(agentDFsList):
            isolationForest_results = []
            testDF_List = []
            for agent_index, agentDF in enumerate(dfList):
                tempTrain = agentDF
                if ('keyNum' in tempTrain):
                    clf = IsolationForest(contamination=0.03).fit(tempTrain)
                    outliers = clf.predict(tempTrain)
                    tempTrain["out"] = outliers
                    plt.scatter("state", "total_reward", c="cyan", alpha=0.005, data = tempTrain[tempTrain["out"]==1])
                    plt.scatter("state", "total_reward", alpha=0.45, c="g", data = tempTrain[tempTrain["out"]==-1])
                    plt.scatter("state", "total_reward", c="r", alpha=0.75, data = tempTrain[tempTrain['keyNum']>0])
                    plt.title(db.agent_names[index] + " " + df_names[df_index] + " " + target_names[target_index])
                    filename = "IsolationForestOutlierDetection" + db.agent_names[index] + df_names[df_index] + target_names[target_index] + ".png"
                    plt.savefig(filename)
                    plt.clf()
                    plt.close()

############################################################################################################
# Anomoly Detection - based on guest speaker's presentation
    if args.vis:
        dfList = [db.basic_num_df, db.df_num_slimmed, db.df_num_full]
        df_names = ["BasicDF", "SlimmedDF", "FullDF"]
        targets = ['agentNum']
        target_names = ["Key States", "Agent Number"]

    for target_index, target in enumerate(targets):

        # Need to call Gaussian Plotting Here
        x_list = ['episode_reward', 'epoch_reward', 'total_reward',
        'episode_step', 'state', 'to_pill_three', 'to_pill_four', 'to_red_ghost',
        'to_pink_ghost', 'importance',
        'to_pill_mean', 'to_top_pills_mean', 'to_bottom_pills_mean',
        'to_ghosts_mean', 'agentNum']
        y_list = ['action', 'reward', 'lives', 'to_pill_one',
        'to_pill_two', 'to_blue_ghost', 'to_orange_ghost', 'importance',
        'to_pill_mean', 'to_top_pills_mean', 'to_bottom_pills_mean',
        'to_ghosts_mean', 'diff_to_red', 'diff_to_orange', 'diff_to_blue',
        'diff_to_pink', 'isDBG']
#
#        folderName = '/Users/byrdsmyth/Documents/School/Classes/CPTS570/Project/codebase/GroundTruth' + targets[target_index]
#        if not os.path.exists(folderName):
#            os.makedirs(folderName)
#        print("HERE1")
#        print(dfList)
#        for index, agentDF in enumerate(dfList):
#            for x in x_list:
#                for y in y_list:
#                    if (x in agentDF):
#                        if (y in agentDF):
#                            if (target in agentDF):
#
#                                if (target == 'isKeyOrContext'):
#                                    plt.scatter(agentDF[x],agentDF[y], c=agentDF[target], alpha=0.5)
#                                    temp = agentDF[agentDF[target] > 0]
#                                    plt.scatter(temp[x],temp[y], c='r')
#                                else:
#                                    temp0 = agentDF[agentDF['agentNum'] == 0]
#                                    temp1 = agentDF[agentDF['agentNum'] == 1]
#                                    temp2 = agentDF[agentDF['agentNum'] == 2]
#                                    temp3 = agentDF[agentDF['agentNum'] == 3]
#                                    temp4 = agentDF[agentDF['agentNum'] == 4]
#                                    plt.scatter(temp0[x],temp0[y], alpha=0.3, s=10, marker=3, label = db.agent_names[0])
#                                    plt.scatter(temp1[x],temp1[y], alpha=0.4, s=9, marker=3, label = db.agent_names[1])
#                                    plt.scatter(temp2[x],temp2[y], alpha=0.5, s=8, marker=3, label = db.agent_names[2])
#                                    plt.scatter(temp3[x],temp3[y], alpha=0.6, s=7, marker=3, label = db.agent_names[3])
#                                    plt.scatter(temp4[x],temp4[y], alpha=0.7, s=6, marker=3, label = db.agent_names[4])
#                                plt.xlabel(x)
#                                plt.ylabel(y)
#                                plt.title(df_names[index] + " Ground truth " + targets[target_index])
#                                filename = "groundTruth" + df_names[index] + targets[target_index] + x + y + ".png"
#                                plt.legend(loc="best")
#                                filePath = os.path.join(folderName, filename)
#                                plt.savefig(filePath)
#                                plt.close()
                                
    
    target = 'isKeyOrContext'
    print("HERE1")
    print(db.agentNumeric_FullDFs)
    for index, agentDF in enumerate(dfList):
        folderName = '/Users/byrdsmyth/Documents/School/Classes/CPTS570/Project/codebase/GroundTruth' + db.agent_names[index]
        if not os.path.exists(folderName):
            os.makedirs(folderName)
            
#        plt.scatter(agentDF['state'],agentDF['importance'], alpha=0.3, s=10, marker=3, label = "Non-Key")
#        temp = agentDF[agentDF['isKeyOrContext'] > 0]
#        plt.scatter(temp['state'],temp['importance'], alpha=0.7, s=6, marker=3, label = "Key")
#        plt.xlabel("Time Step")
#        plt.ylabel("Importance")
#        plt.title(db.agent_names[index] + " Importance Vs Key States")
#        filename = "groundTruth" + db.agent_names[index] + "IMPORTANCE" + ".png"
#        plt.legend(loc="best")
#        filePath = os.path.join(folderName, filename)
#        plt.savefig(filePath)
#        plt.close()
        for x in x_list:
            for y in y_list:
                if (x in agentDF):
                    if (y in agentDF):
                        if (target in agentDF):
                            plt.scatter(agentDF[x],agentDF[y], alpha=0.3, s=10, marker=3, label = db.agent_names[0])
                            temp = agentDF[agentDF[target] > 0]
                            plt.scatter(temp[x],temp[y], alpha=0.7, s=6, marker=3, label = "Key States")
                            plt.xlabel(x)
                            plt.ylabel(y)
                            plt.title(db.agent_names[index] + " Ground truth Key States")
                            filename = "groundTruth" + db.agent_names[index] + "keyStates" + x + y + ".png"
                            plt.legend(loc="best")
                            filePath = os.path.join(folderName, filename)
                            plt.savefig(filePath)
                            plt.close()

            
