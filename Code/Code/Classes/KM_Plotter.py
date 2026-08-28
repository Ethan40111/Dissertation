from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

class KM_Plotter():
    
    def __init__(self,risk_labels_df,num_curves=10):
        self.risk_labels_df = risk_labels_df
        self.num_curves = num_curves
        self.survival_functions = []
        self.survival_functions_temp = []
        self.optimal_separation = -1 #to be initialised
        
    def makeKMPlot(self,plot=True):
        km_data = pd.DataFrame({
            'time': self.risk_labels_df['Survival_Time'],
            'event': self.risk_labels_df['Event_Status'].astype(int), #0 - censored
            'risk_label': self.risk_labels_df['Risk_Label']
        })
        if plot:
            plt.figure(figsize=(14,8))
        kmf = KaplanMeierFitter()

        #Define colours for each risk group
        color = ['#2ecc71', '#27ae60', '#f1c40f', '#f39c12', '#e67e22', 
                '#d35400', "#a55e56", '#c0392b', '#8e44ad', '#2c3e50']

        risk_labels_sorted = [f'EL{i}' for i in range(1,self.num_curves+1)]

        for i,label in enumerate(risk_labels_sorted):
            
            mask = km_data['risk_label'] == label
            n_patients = mask.sum()

            if n_patients > 0:
                kmf.fit(
                    durations = km_data.loc[mask,'time'],
                    event_observed = km_data.loc[mask,'event'],
                    label = f'{label} (n={n_patients})'
                    )
                
                self.survival_functions.append(kmf.survival_function_)

                print(f"appending curve for risk label {i}")
                if plot:
                    ax = kmf.plot_survival_function(
                        color=color[i],
                        linewidth=2.5,
                        alpha=0.8 #changing increases length of confidence region but decreases size of box 
                    )
                    plt.xlim(0, 120)

        self.survival_functions_temp = self.survival_functions
        
        #take each survival function in our survival functions list and join them accordingly
        #create timeline from 0 to 150 with step size 1
        timeline = pd.DataFrame({'timeline': np.arange(0, 151, 1)})

        #temp is a list of DataFrames (one per risk group), each with time as index and survival probability as value
        #we want to left join each survival function onto the timeline by the index (time)
        merged = timeline.copy()
        print("Survival functions df is: ")
        print(self.survival_functions)
        print("Length of temp survival functions is: ", len(self.survival_functions_temp))
        print(f"Length of survival functions is: {len(self.survival_functions)}")
        for i, survival_function in enumerate(self.survival_functions):
            print(f"{i}th enumeration")
            #reset index to get time as a column, rename survival prob column
            surv_df = survival_function.reset_index()
            surv_df.columns = ['timeline', f'{i+1}']
            merged = merged.merge(surv_df, on='timeline', how='left')
        #cut off timeline values beyond 120
        merged = merged[merged['timeline'] <= 120].reset_index(drop=True)
        self.survival_functions = merged
        self.optimal_separation = 1/self.num_curves
        print(f"length of survival functions is: {len(self.survival_functions)}")
        
   
    def getKMScore(self, missing_penalty=1):
        #penalise incomplete curves
        #score baseed off of distance between curves relative to 1/(number of curves)
        #get all pairwise combinations of columns
        #labels should have order and there should be no overlap 
        print("Update 16")
        print(f"Length of survival functions: {len(self.survival_functions)}")
        print(f"survival functions columns: {self.survival_functions.columns}")
        #from 0 to num_curves-1, get consecutive pairs of columns
        consecutive_pairs = [(i, i+1) for i in range(1,len(self.survival_functions.columns)-1)]
        parallelism_scores_df = pd.DataFrame()
        pair_scores = []
        missing_curves = self.num_curves - len(self.survival_functions.columns) + 1
        if missing_curves > 0: #don't want missing curves
            return 1000
        
        #ensure ranking between risk labels is conserved and there is no overlap 
        for col_index in range(1, len(self.survival_functions.columns)-1):
            for row in range(len(self.survival_functions)):
                #if value for row and col_index is higher than the same row for all proceeding columns, penalise
                value = self.survival_functions.iloc[row, col_index]
                if not pd.isna(value) and not np.isinf(value):
                    for proceeding_col_index in range(col_index+1, len(self.survival_functions.columns)-1):
                        proceeding_value = self.survival_functions.iloc[row, proceeding_col_index]
                        if not pd.isna(proceeding_value) and not np.isinf(proceeding_value):
                            if value <= proceeding_value:
                                return 1000 #worst possible score
                    
        for consecutive_pair in consecutive_pairs:
            pair_differences = []
            print(f"Consecutive pair being compared is: {consecutive_pair}")
            for row in range(len(self.survival_functions)):
                
                value_1 = self.survival_functions.iloc[row, consecutive_pair[0]]
                value_2 = self.survival_functions.iloc[row, consecutive_pair[1]] # higher value
                if not pd.isna(value_1) and not np.isinf(value_1) and not pd.isna(value_2) and not np.isinf(value_2):
                    difference =  abs((value_2 - value_1 - self.optimal_separation))
                else:
                    difference = missing_penalty
                print(f"difference: {difference}")
                pair_differences.append(difference)

            if len(pair_differences) == 0:
                average_difference = missing_penalty
            else:
                average_difference = np.mean(pair_differences)

            parallelism_scores_df[str(consecutive_pair)] = [average_difference]
            pair_scores.append(average_difference)

        
        if parallelism_scores_df.empty or len(pair_scores) == 0:
            print("scores are empty, returning 1000")
            return 1000 #worst possible predictions
        else:
            average = np.mean(pair_scores)
            if np.isnan(average):
                print("average is NaN, returning worst score of 1000")
                return 1000
            else:

                return average

