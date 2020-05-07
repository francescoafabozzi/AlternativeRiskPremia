import numpy as np
import pandas as pd
import pickle
import os

import warnings
warnings.simplefilter('ignore',category=RuntimeWarning)



#################################################################################

# Load Data
def load_data(file_path_metadata, file_path_prices):
    metadata = pd.read_excel(file_path_metadata)
    prices = pd.read_excel(file_path_prices)

    # clean metadata
    metadata.index = metadata.iloc[:, 0]
    metadata = metadata.iloc[:, 1:]
    metadata = metadata.transpose()

    # cleaning price data
    prices.columns = prices.iloc[1, :]
    prices = prices.iloc[2:, :]
    prices.index = pd.to_datetime(prices.iloc[:,0])
    prices = prices.iloc[:, 1:]
    prices = prices.astype(float)
    # get rid of weekdays
    prices = prices.loc[[i for i in prices.index if i.weekday() < 5], :]

    # split index returns from cash conversions
    cash_conversions = prices[['USD', 'EUR', 'JPY', 'GBP', 'CHF']].copy()
    prices = prices.iloc[:, :-6]

    return {'Prices': prices, 'metadata': metadata, 'Cash': cash_conversions}


def find_runs(return_columns):
    ret_test = return_columns.dropna().copy()
    last_ind = ret_test.shape[0] - 1
    zeros = np.where(ret_test == 0)[0]
    if len(zeros) == 0:
        return False
    nums = sorted(set(zeros))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    conseq = list(zip(edges, edges))
    runs = [i for i in conseq if i[0] != i[1]]

    if len(runs) == 0:
        return False

    if 0 not in runs[0] and last_ind not in runs[-1]:
        return False

    elif 0 in runs[0] and last_ind in runs[-1]:
        front = (runs[0][1] - runs[0][0]) + 1
        middle = [(i[1] - i[0]) + 1 for i in runs[1:-1]]
        end = (runs[-1][1] - runs[-1][0]) + 1
        return ['Both', {'Front': front, 'Middle': middle, 'End': end}]


    elif 0 in runs[0] and last_ind not in runs[-1]:
        front = (runs[0][1] - runs[0][0]) + 1
        middle = [(i[1] - i[0]) + 1 for i in runs[1:]]
        end = 0
        return ['Front', {'Front': front, 'Middle': middle, 'End': end}]

    elif 0 not in runs[0] and last_ind in runs[-1]:
        front = 0
        middle = [(i[1] - i[0]) + 1 for i in runs[:-1]]
        end = (runs[-1][1] - runs[-1][0]) + 1
        return ['End', {'Front': front, 'Middle': middle, 'End': end}]

###############################################################################

###############################################################################
# define functions to fix front and ends if runs are significant

def fix_front(x):
    new_return_start = x.dropna().replace(0, np.nan).dropna().index[0]
    price_start = prices[:new_return_start].index[-2]
    return price_start

def fix_end(x):
    new_return_end = x.dropna().replace(0, np.nan).dropna().index[-1]
    return new_return_end



##################################################################################


class Preprocessing:
    def __init__(self,file_path_metadata,file_path_prices, flat_thresh,redun_thresh):
        '''
        file_path_metadata:
            path to the excel file containing the metadata

        file_path_prices:
            file path the the excel file containing all investable indices

        flat_thresh:
            the quantile used to determine whether the are flat fillings at beginning/end

        redun_thresh:
            the correlation threshold used to determine if indices are redundant

        '''
        data_dict = load_data(file_path_metadata, file_path_prices)
        
        self.raw_prices = data_dict['Prices']
        self.metadata = data_dict['metadata']
        self.cash_conversions = data_dict['Cash']
        self.corrections = self.get_flat_corrections(self.raw_prices,flat_thresh)
        self.new_prices = self.get_new_prices()
        self.binary_price_mat = self.new_prices.notnull().astype(int)
        self.compare_df = self.get_comparison_df()
        self.returns = self.get_cash_conversions()
        ####################################################
        self.returns_new = self.drop_redundant_indices(redun_thresh)
        self.gross_returns = self.get_gross_returns()
        self.net_returns = self.get_net_returns()
        self.dropped_indices = [i for i in self.returns.columns if i not in self.returns_new]
        ####################################################
        self.new_metadata = self.get_new_metadata()

        
    def get_net_returns(self):
        external_costs = self.metadata.EXTERNAL_COSTS.copy()
        returns = self.returns.copy()
        for index,cost in external_costs.iteritems():
            returns[index] -= cost/252.

        return returns


    def get_gross_returns(self):
        internal_costs = self.metadata.INTERNAL_COSTS.copy()
        returns= self.returns.copy()
        for index,cost in internal_costs.iteritems():
            returns[index] += cost/252.

        return returns

    def get_new_prices(self):
        corrections = self.corrections.copy()
        new_p = self.raw_prices.copy()
        for i in corrections.iterrows():
            index = i[0]
            end = i[1]['New End']
            start = i[1]['New Start']
            if end != False:
                end += pd.offsets.Day(1)
                new_p.loc[end:, index] = np.nan

            if start != False:
                start -= pd.offsets.Day(1)
                new_p.loc[:start, index] = np.nan

        return new_p

    def get_cash_conversions(self):
        cash = self.cash_conversions.copy()
        prices = self.new_prices.copy()
        metadata = self.metadata.copy()
        returns = np.log(prices / prices.shift(1))
        converted_returns = returns.copy()
        cash_returns = np.log(cash / cash.shift(1))
        for i in ['USD', 'EUR', 'JPY', 'GBP', 'CHF']:
            convert = metadata[metadata[i] == 1].index.tolist()
            for l in convert:
                check = pd.concat([returns[l].dropna(), cash_returns[i]], axis=1, join='inner')
                converted_returns[l] = check.iloc[:, 0] - check.iloc[:, 1]

        return converted_returns



    def get_flat_corrections(self,prices, quantile):
        '''
        We need prices in order to find where we need to fix the start of the price series.

        Thus, the output of this function returns corrections to the price dates... NOT RETURN DATES.
        '''

        returns = np.log(prices) - np.log(prices.shift())
        returns = returns.iloc[1:, :]

        corrections = pd.DataFrame(columns=['New Start', 'New End'])
        for i in returns.columns:
            x = returns.loc[:, [i]]
            runs = find_runs(x)
            if runs == False:
                correct = pd.DataFrame({'New Start': [False], 'New End': [False]}, index=[i])
                corrections = pd.concat([corrections, correct], axis=0)

            elif runs[0] == 'Front':
                vals = runs[1]
                front = vals['Front']
                middle = vals['Middle']
                if len(middle) <= 1 or front >= np.quantile(middle, quantile):
                    new_start = self.fix_front(x)
                    correct = pd.DataFrame({'New Start': [new_start], 'New End': [False]}, index=[i])
                    corrections = pd.concat([corrections, correct], axis=0)
                else:
                    correct = pd.DataFrame({'New Start': [False], 'New End': [False]}, index=[i])
                    corrections = pd.concat([corrections, correct], axis=0)

            elif runs[0] == 'End':
                vals = runs[1]
                end = vals['End']
                middle = vals['Middle']
                if len(middle) <= 1 or end >= np.quantile(middle, quantile):
                    new_end = self.fix_end(x)
                    correct = pd.DataFrame({'New Start': [False], 'New End': [new_end]}, index=[i])
                    corrections = pd.concat([corrections, correct], axis=0)
                else:
                    correct = pd.DataFrame({'New Start': [False], 'New End': [False]}, index=[i])
                    corrections = pd.concat([corrections, correct], axis=0)

            elif runs[0] == 'Both':
                vals = runs[1]
                front = vals['Front']
                end = vals['End']
                middle = vals['Middle']
                if len(middle) <= 2:
                    new_start = self.fix_front(x)
                    new_end = self.fix_end(x)
                elif front >= np.quantile(middle, quantile) and end >= np.quantile(middle, quantile):
                    new_start = self.fix_front(x)
                    new_end = self.fix_end(x)

                elif front >= np.quantile(middle, quantile) and end < np.quantile(middle, quantile):
                    new_start = self.fix_front(x)
                    new_end = False

                elif front < np.quantile(middle, quantile) and end >= np.quantile(middle, quantile):
                    new_start = False
                    new_end = self.fix_end(x)

                else:
                    new_end = False
                    new_start = False

                correct = pd.DataFrame({'New Start': [new_start], 'New End': [new_end]}, index=[i])
                corrections = pd.concat([corrections, correct], axis=0)

        return corrections

    def fix_front(self,x):
        new_return_start = x.dropna().replace(0, np.nan).dropna().index[0]
        price_start = self.raw_prices[:new_return_start].index[-2]
        return price_start

    def fix_end(self,x):
        new_return_end = x.dropna().replace(0, np.nan).dropna().index[-1]
        return new_return_end

    @staticmethod
    def find_runs(return_columns):
        ret_test = return_columns.dropna().copy()
        last_ind = ret_test.shape[0] - 1
        zeros = np.where(ret_test == 0)[0]
        if len(zeros) == 0:
            return False
        nums = sorted(set(zeros))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        conseq = list(zip(edges, edges))
        runs = [i for i in conseq if i[0] != i[1]]

        if len(runs) == 0:
            return False

        if 0 not in runs[0] and last_ind not in runs[-1]:
            return False

        elif 0 in runs[0] and last_ind in runs[-1]:
            front = (runs[0][1] - runs[0][0]) + 1
            middle = [(i[1] - i[0]) + 1 for i in runs[1:-1]]
            end = (runs[-1][1] - runs[-1][0]) + 1
            return ['Both', {'Front': front, 'Middle': middle, 'End': end}]


        elif 0 in runs[0] and last_ind not in runs[-1]:
            front = (runs[0][1] - runs[0][0]) + 1
            middle = [(i[1] - i[0]) + 1 for i in runs[1:]]
            end = 0
            return ['Front', {'Front': front, 'Middle': middle, 'End': end}]

        elif 0 not in runs[0] and last_ind in runs[-1]:
            front = 0
            middle = [(i[1] - i[0]) + 1 for i in runs[:-1]]
            end = (runs[-1][1] - runs[-1][0]) + 1
            return ['End', {'Front': front, 'Middle': middle, 'End': end}]


    def get_comparison_df(self):
        original = []
        for i in self.raw_prices.iteritems():
            start = i[1].dropna().index[0]
            end = i[1].dropna().index[-1]
            original.append([i[0],start,end])

        original_df = pd.DataFrame(original,
                                   columns=['Index',
                                            'Original Start',
                                            'Original End']).set_index('Index')

        compare_df=pd.concat([original_df,
                              self.corrections],axis=1).loc[:,['Original Start','New Start',
                                                               'Original End','New End']]

        return compare_df

    def drop_redundant_indices(self, thresh):
        rets = self.returns.copy()
        metadata = self.metadata.copy()

        data_dict = {}
        for i in metadata.INSTITUTION.unique():
            data_dict.update({i: rets.loc[:, [metadata.index[j] \
                                              for j in range(metadata.shape[0]) if \
                                              metadata.INSTITUTION[j] == i]]})

        redundant_dict = {}
        for k in metadata.INSTITUTION.unique():
            bank_data = data_dict[k].copy()
            bank_corr = bank_data.corr()
            high_corr = np.where(bank_corr.abs() >= thresh)
            high_corr_list = []
            for i in range(len(high_corr[0])):
                if high_corr[0][i] != high_corr[1][i]:
                    if (high_corr[1][i], high_corr[0][i]) not in high_corr_list:
                        high_corr_list.append((high_corr[0][i], high_corr[1][i]))

            corr_df = pd.DataFrame()
            for j in high_corr_list:
                ind1 = bank_corr.columns[j[0]]
                ind2 = bank_corr.columns[j[1]]
                ind1_len = bank_data.loc[:, ind1].dropna().shape[0]
                ind2_len = bank_data.loc[:, ind2].dropna().shape[0]
                check = bank_data.loc[:, [ind1, ind2]].dropna()
                corr = check.corr().iloc[0, 1]
                meta_sub = metadata.loc[[ind1, ind2], :]
                difference = (meta_sub.iloc[0, :] == meta_sub.iloc[1, :]).replace(True, np.nan).dropna().index
                meta_add = (meta_sub.iloc[0, :] == meta_sub.iloc[1, :]).to_frame().transpose()
                meta_add = meta_add.loc[:, ['OBJECTIVE', 'STYLE', 'ASSET_CLASS', 'REGION',
                                            'INDEX_DESCRIPTION', 'ADJ_LIVE_START_DATE',
                                            'EXTERNAL_COSTS', 'INTERNAL_COSTS']]
                try:
                    difference = difference.drop('INDEX_NAME').tolist()
                except:
                    difference = difference.tolist()
                add_df = pd.DataFrame({
                    'Index 1': [ind1],
                    'Index 2': [ind2],
                    'Correlation': [corr],
                    'Index 1 Length': [ind1_len],
                    'Index 2 Length': [ind2_len]
                    # 'Difference': [difference]
                })
                add_df = pd.concat([add_df, meta_add], axis=1)
                corr_df = pd.concat([corr_df, add_df], axis=0)

            redundant_dict.update({k: corr_df})

        all_replacements = pd.concat([v for i, v in redundant_dict.items()], axis=0)

        # get drops
        drop1 = all_replacements[all_replacements['Index 2 Length'] > \
                                 all_replacements['Index 1 Length']]['Index 1']
        drop2 = all_replacements[all_replacements['Index 1 Length'] >= \
                                 all_replacements['Index 2 Length']]['Index 2']
        drops = list(np.unique(drop1)) + list(np.unique(drop2))
        new_rets = rets.drop(drops, axis=1)
        dropped_rets = rets[drops]

        check = []
        for index, values in dropped_rets.iteritems():
            corrs = new_rets.corrwith(values)
            if corrs[corrs >= thresh].shape[0] < 1:
                check.append(index)

        new_rets = pd.concat([new_rets, rets[check]], axis=1)

        #dropped = [i for i in rets.columns if i not in new_rets.columns]
        #self.testing=all_replacements[all_replacements['Index 1'] in dropped]

        return new_rets
    
    
    def redundancy_mat(self):
        binary_redun= pd.DataFrame([1 for i in range(self.raw_prices.shape[1])],
                                   index=data_class.raw_prices.columns.tolist()).transpose()
        binary_redun.loc[0,self.dropped_indices] = 0
        binary_redun.index = ['Used']
        return binary_redun
    
    
    @staticmethod
    def check_replacements(replacement_df):
        backfill = []
        switch = []
        for ind,vals in replacement_df.iterrows():
            rep_start = vals['Replacement Start']
            rep_end = vals['Replacement End']
            old_start = vals['Dropped Start'] 
            old_end = vals['Dropped End']

            if old_start < rep_start and old_end > rep_end:
                backfill.append('Both')
            elif old_start < rep_start:
                backfill.append('Backfill')
            elif old_end > rep_end:
                backfill.append('Forward Fill')
            else:
                backfill.append(False)

        replacement_df.loc[:,'Back/Forward Fill'] = backfill
        
        return replacement_df
    
    
    
    def get_replacements(self):
        
        groupings = ['OBJECTIVE','STYLE','ASSET_CLASS',
                     'REGION','INDEX_DESCRIPTION']
        
        dropped=self.dropped_indices
        dropped_rets = self.returns[dropped].copy()
        kept_rets = self.returns.drop(dropped,axis=1).copy()
        metadata = self.metadata
        
        
        replacement_df = pd.DataFrame()
        for i in dropped:
            st=dropped_rets[i].dropna().index[0]
            end =dropped_rets[i].dropna().index[-1]
            bank = metadata.loc[i,'INSTITUTION']
            all_inds = metadata[metadata['INSTITUTION'] ==  bank].index.tolist()
            all_inds = [i for i in all_inds if i in kept_rets.columns.tolist()]
            all_rets = kept_rets[all_inds]    
            replacement =all_rets.corrwith(dropped_rets[i]).idxmax()
            rep_start=all_rets[replacement].dropna().index[0]
            rep_end=all_rets[replacement].dropna().index[-1]
            diff=metadata.loc[i,groupings] == metadata.loc[replacement,groupings]
            meta_diff=[groupings[i] for i,v in enumerate(diff) if v == False]
        
            df= pd.DataFrame({
                'Replacement':[replacement],
                'Dropped Start':[st],
                'Replacement Start':[rep_start],
                'Dropped End':[end],
                'Replacement End':[rep_end],
                'Metadata Difference': [meta_diff]
            },index=[i])

            replacement_df=pd.concat([replacement_df,df],axis=0)
            
        return self.check_replacements(replacement_df)
        
        
    def get_new_metadata(self):
        new_metadata = self.metadata.copy()
        price_mat = self.binary_price_mat.copy()
        starting = []
        end = []
        for i in price_mat.columns:
            p = price_mat[[i]]
            new_start=p.replace(0,np.nan).dropna().index[0]
            new_end = p.replace(0,np.nan).dropna().index[-1]
            end.append(new_end)
            if new_start == new_metadata.loc[i,'ADJ_LIVE_START_DATE']:
                starting.append(np.nan)
            else:
                starting.append(new_start)
                
        new_metadata.loc[:,'NEW_START'] = starting
        new_metadata.loc[:,'NEW_END'] = end
        
        return new_metadata
    
    
    
    
## Function to save final output    
def save_output(data):
    replacements = data.get_replacements()
    replacements.to_csv('Output/replacements.csv')
    data.binary_price_mat.to_csv('Output/binary_price_mat.csv')
    data.new_metadata.to_csv('Output/new_metadata.csv')
    return 'Files saved in Output Folder'
        
    







