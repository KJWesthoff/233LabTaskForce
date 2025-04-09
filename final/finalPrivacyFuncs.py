
from operator import eq

import numpy as np
import pandas as pd
from pandas._config import display


def get_equivalence_classes(df, quasi_identifiers):
    """
    Generate equivalence classes based on the given quasi-identifiers.

    Parameters:
        df (pd.DataFrame): The input dataset.
        quasi_identifiers (list): List of quasi-identifier column names.

    Returns:
        dict: A dictionary where keys are tuples of quasi-identifier values, 
              and values are lists of corresponding row indices.
    """
    equivalence_classes = df.groupby(quasi_identifiers).apply(lambda x: x.index.tolist(), include_groups=False).to_dict()
    return equivalence_classes



def k_anonymity(df, quasi_identifiers):
    """
    Check k-anonymity for a given DataFrame.

    Parameters:
        df (pd.DataFrame): The input dataset.
        quasi_identifiers (list): List of quasi-identifier column names.

    Returns:
        integer: k-anon the k-anonymity score for the QI.
        pd.DataFrame: Grouped counts of quasi-identifier combinations.
    """
    grouped = df.groupby(quasi_identifiers).size().reset_index(name="count")
    k_anon = grouped["count"].min()
    return k_anon, grouped


def distinct_l_diversity(df, quasi_identifiers, sensitive_attribute):
    """
    Check distinct l-diversity: Ensures each group has at least 'l' distinct sensitive values.

    Parameters:
        df (pd.DataFrame): The input dataset.
        quasi_identifiers (list): List of quasi-identifier column names.
        sensitive_attribute (str): The sensitive attribute column.

    Returns:
        integer: score for distinct l-diversity.
        pd.DataFrame: Grouped data with distinct value counts.
    """
    
    ## Group the df sequentially by the quasi identifiers, then select the sensitive attribute and count how many unique values it has
    diversity_scores = df.groupby(quasi_identifiers)[sensitive_attribute].nunique() 
    distinct_l = diversity_scores.min()
    result_df = diversity_scores.reset_index(name=f"n unique {sensitive_attribute}")
    return distinct_l, result_df


def entropy_l_diversity(df, quasi_identifiers, sensitive_attribute):
    """
    Check entropy l-diversity: Calculates the l-diversity for the table using log2.

    Parameters:
        df (pd.DataFrame): The input dataset.
        quasi_identifiers (list): List of quasi-identifier column names.
        sensitive_attribute (str): The sensitive attribute column.

    Returns:
        l_diversity: Calculated l-diversity for the whole table (the minimum found for all equiv. class)
        result_df: pd.DataFrame Grouped data with entropy values for each quasi identifier.
    """
    
    def entropy(group):
        """
        Sub function to calculate entropy, per entry 
        Parameters: 
            group, output from df.groupby (df like object) the function operates on each row
            and calculates the entropy
        Returns:
            The summed up entropy and 'reversed' l value for each equivalence class
        """
        probs = group[sensitive_attribute].value_counts(normalize=True) ## probabilities using normalization in one go
        #print(probs)
        log2entropy = -np.sum(probs * np.log2(probs))  # Shannon entropy
        return 2**log2entropy # Reverse the log to get l

    ## crounch the l-scores for each equivalence class
    diversity_scores = df.groupby(quasi_identifiers).apply(entropy, include_groups=False)
    l_diversity = diversity_scores.min()  # Entropy threshold
    result_df = diversity_scores.reset_index(name=f"entropy l {sensitive_attribute}")
    return l_diversity, result_df


def recursive_l_diversity(df, quasi_identifiers, sensitive_attribute, verbose = False):
    """
    Calculate recursive (c,l)-diversity: Ensures the top sensitive value is not too dominant.

    Parameters:
        df (pd.DataFrame): The input dataset.
        quasi_identifiers (list): List of quasi-identifier column names.
        sensitive_attribute (str): The sensitive attribute column.
        verbose: defaults to False, and the function outputs worst cases at max l
                ,if False it will output c's for all l 

    Returns:
        c for l: recursive (c,l)-diversity
        pd.DataFrame: Grouped data with boolean values indicating compliance.
    """

    def recursive_diversity(group):
        """
        Sub function to calculate c, per entry 
        Parameters: 
            group, output from df.groupby (df like object) the function operates on each row
            and calculates the entropy
        Returns:
            The c for each group (equivalence class)
        """
        ## Handy dandy value_counts counts the unique value of sensitive attribute, and orders high to low
        value_counts = group[sensitive_attribute].value_counts().values
        #print(value_counts)

        c = value_counts[0] / np.sum(value_counts[1:])  # Top value vs. rest
        return(c)
    

    def recursive_diversity_verbose(group):
        """
        Sub function to calculate c, per entry 
        Parameters: 
            group, output from df.groupby (df like object) the function operates on each row
            and calculates the entropy
        Returns:
            The c for each group (equivalence class)
        """
        ## Handy dandy value_counts counts the unique value of sensitive attribute, and orders high to low
        value_counts = group[sensitive_attribute].value_counts().values
        #print(value_counts)
        ## Calculate c for all l's         
        l_max = len(value_counts)
        ## calculate all the c's
        all_c = [value_counts[0] / np.sum(value_counts[l:]) for l in range(0,l_max)]        
        all_c_df = pd.DataFrame({'l':range(1,l_max+1), 'c': all_c})

        
        return(all_c_df)
    




    ## Run the diversity calculation on on all eqivalence classes and return worst case
    if verbose:
        diversity_scores = df.groupby(quasi_identifiers).apply(recursive_diversity_verbose, include_groups=False)
        worst_case_c = diversity_scores.loc[diversity_scores['l'] == 2, 'c'].max()
    else:
        diversity_scores = df.groupby(quasi_identifiers).apply(recursive_diversity, include_groups=False)
        worst_case_c = max(diversity_scores)
    

    result_df = diversity_scores #.reset_index(name="recursive_l_diversity")
    return worst_case_c, result_df

# Your solution goes here
def EMD_ordered_distance(df, quasi_identifiers, sensitive_attribute, sensitive_values_order):
    """
    Calculate Earth Movers Distance.

    Parameters:
        df (pd.DataFrame): The input dataset.
        quasi_identifiers (list): List of quasi-identifier column names.
        sensitive_attribute (str): The sensitive attribute column.
        sensitive_values_order: List The order of sensitive values i.e. the weights from most important to least 

    Returns:
        float t: the worst case (max) earth movers distance
        pd.DataFrame equivalence_classes: the data grouped by Equivalence classes, with a column added for EMD 
    """


    
    if sensitive_values_order:

        ## calculate p for the whole table sorted by 'order'
        table_sens_vals = df[sensitive_attribute].value_counts() 
        table_sorted = table_sens_vals.reindex(sensitive_values_order, fill_value=0)
        table_counts = sum(table_sorted)
        p = table_sorted/table_counts
        
        #print(p)
        
        def calc_EMD(EC):
            

            V = len(sensitive_values_order)
            eq_sens_vals = EC.value_counts()
            eq_sorted = eq_sens_vals.reindex(sensitive_values_order, fill_value=0)
            eq_counts = sum(eq_sorted)
            q = eq_sorted/eq_counts

            ## Calculate the residue, what needs to be moved at each step
            r = p-q 
            
            ## Cumulative sum of effort done at each step keeping direction so some of them cancels out or are reduced
            cs = np.cumsum(r)

            ## Calculate absolute effort at each step
            abs = np.abs(cs)
            
            ## calculate the total work done
            total_work = np.sum(abs)

            EC_EMD = total_work/(V-1)
            return EC_EMD


    if not sensitive_values_order:
        ## calculate p for the whole table sorted by 'order'
        
        ## Do the normalization in one go
        table_sens_vals = df[sensitive_attribute].value_counts(normalize=True) 
        
        #table_counts = sum(table_sorted)
        p = table_sens_vals
     

        ## Do the same on the EC's
        def calc_EMD(EC):
            eq_sens_vals = EC.value_counts(normalize=True)
            q = eq_sens_vals

            ## Calculate the residue, what needs to be moved at each step
            r = p-q 
            

            ## Calculate absolute effort at each step
            abs = np.abs(r)
            
            ## calculate the total work done
            total_work = np.sum(abs)

            EC_EMD = total_work/2
            return EC_EMD


    equivalence_classes = df.groupby(quasi_identifiers).agg(EMD=(sensitive_attribute,calc_EMD))
    t = max(equivalence_classes.EMD)
    return t,equivalence_classes


    
def delta_presence(df_identities, df_delta, quasi_identifiers):
    
    """
    Calculate Delta presence between an identified data and an additional delta dataset published.

    Parameters:
        df_identities (pd.DataFrame): Input dataset containing the identities.
        df_delta anonymized dataset containing the sensitive values 
        quasi_identifiers (list): List of quasi-identifier column names. (To do: this could be defaulted to all other than idetifier (eg name) in the df_identities)

    Returns:
        out: pd.DataFrame equivalence_classes: the original df_identities with delta calculated for each entry
        deltas: tuple: delta (min,max)
    """

    out = df_identities.copy()  # copy the original DataFrame
    
    # Define Equivalence classes
    delta_ECs = df_delta.groupby(quasi_identifiers).size()

    # Initialize delta column
    out["delta"] = 0.0

    # Loop over each Equivalence Class (EC) in Identities
    for ec, items in df_identities.groupby(quasi_identifiers):
        
        ec = ec[0] if len(ec) == 1 else ec ## Hackaddi hack get does not accept iterables of length 1 ...

        ##print(f'EC:{ec} ')
        
        # Check if EC exists in df_delta and calculate ping
        ping = delta_ECs.get(ec, 0)  # Use pd get to find the same EC in the delta'ed DataDrame, Default to 0 if EC is not in there
        pong = len(items)
        delta = ping / pong
        
        # Assign the delta value to the corresponding rows in 'out', update on each iteration of the for loop
        out.loc[items.index, "delta"] = delta

    # Return the updated DataFrame and the min/max delta values
    deltas = (out['delta'].min(), out['delta'].max())
    
    return out, deltas


def Mondrian_allowable_dims(table_partition, quasi_identifiers, k, cut_choice_fcn):
    """A helper function that determines the list of dimensions with allowable cuts.
    
    Parameters
    ----------
    table_partition : pd.DataFrame
        The partition on which to determine the dimensions with allowable cuts
    quasi_identifiers : list
        A list containing the set of quasi-identifiers (or dimensions)
    k : int
        The desired k
    cut_choice_fcn : types.FunctionType (lambda data, k: number)
        A function pointer to the cut value selection strategy
    
    Returns
    -------
    allowable_dims : list
        A list containing the set of dimensions with allowable cuts.
    """
    
    # We don't know which dimensions will have allowable cuts yet, so we initialize allowable_dims to an empty list
    allowable_dims = []
    for dim_name in quasi_identifiers:
        # For this dimension, the values are
        dim_values = table_partition.loc[:,dim_name]
        # and the cut value is
        dim_boundry_cut = cut_choice_fcn(dim_values.to_list(), k)
        lhs = table_partition.loc[table_partition.loc[:,dim_name] <= dim_boundry_cut,:]
        rhs = table_partition.loc[table_partition.loc[:,dim_name] > dim_boundry_cut,:]
        if lhs.shape[0] >= k and rhs.shape[0] >= k:
            allowable_dims.append(dim_name)
    return allowable_dims

def Mondrian(table_in, quasi_identifiers, k, dim_choice_fcn, cut_choice_fcn):
    """The Mondrian algorithm implementation.
    
    Parameters
    ----------
    table_in : pd.DataFrame
        The input table to be generalized
    k : int
        The desired k
    dim_choice_fcn : types.FunctionType (lambda partition, allowable_dims: string)
        A function pointer to the dimension selection strategy
    cut_choice_fcn : types.FunctionType (lambda data, k: number)
        A function pointer to the cut value selection strategy
    
    Returns
    -------
    table_out : pd.DataFrame
        The generalized k-Anonymous table
    partition_boundaries : pd.DataFrame
        A dataframe describing for each partition:
            - the partition boundaries for each quasi-identifier (minimum, maximum), and
            - the partition's final k value
    
    Raises
    ------
    Exception
        If table_in cannot be made k-Anonymous.
    """
    
    if table_in.shape[0] < k:
        # Impossible, we can't achieve k-Anonymity, there aren't enough
        # rows in the table!
        raise Exception('It is impossible to k-Anonymize the input table. There are fewer than k rows in the provided table')
    allowable_dims = Mondrian_allowable_dims(table_in, quasi_identifiers, k, cut_choice_fcn)
    if len(allowable_dims) == 0:
        # In this case, there are no more allowable cuts for this partition.
        # Go through the different attributes and change their values by the boundaries of the partition (i.e., sanitize).
        nr_dims = len(quasi_identifiers)
        # initialize the partitions table
        partition_boundaries = pd.DataFrame(index=np.arange(1), columns=quasi_identifiers + ['k'])
        # get the number of rows for the output table
        nr_rows = table_in.shape[0]
        # initialize the output table
        table_out = pd.DataFrame(index=table_in.index, columns=table_in.columns)
        # In the output table, set the values of the quasi identifiers according to the partition boundaries
        for dim_name in quasi_identifiers:
            # find the boundaries
            ####################
            # If the attribute values are strings rather than numerical, we want the boundaries to include
            # all the values rather than the min and max.
            if len(table_in.loc[:,dim_name]) and isinstance(table_in.loc[:,dim_name].iloc[0], str):
                curr_boundaries = tuple(set(table_in.loc[:,dim_name]))
            else:
                curr_boundaries = (min(table_in.loc[:,dim_name]), max(table_in.loc[:,dim_name]))
            ####################
            # set the boundries of this dimension in the output variable
            partition_boundaries.loc[:,dim_name] = [curr_boundaries]
            # and set all values of this column to the boundaries in the output table
            # display(numpy.tile(partition_boundaries.loc[:,dim_name],(nr_rows,1)))
            table_out.loc[:,dim_name] = [curr_boundaries]
            
        # Now that we've worked all quasi-identifiers, we need to copy the values of the sensitive attributes (untouched).
        for dim_name in table_in.columns.difference(quasi_identifiers):
            table_out.loc[:,dim_name] = table_in.loc[:,dim_name]
        # # Package partition boundaries in a list (to prepare it to be appended to other partition boundaries later)
        # partition_boundaries = [partition_boundaries]
        # finally, the output k for this table is the size of the table
        partition_boundaries.loc[:,'k'] = [table_out.shape[0]]
    else:
        # In this case, there is at least one dimension with an allowable cut.
        # Choose a dimension according to our dimension choice function.
        dim = dim_choice_fcn(table_in, allowable_dims)
        
        # Now we calculate the cut value
        dim_boundry_values = table_in.loc[:,dim]
        dim_boundry_cut = cut_choice_fcn(dim_boundry_values.to_list(), k)
        
        # The left hand side cut
        lhs = table_in.loc[table_in.loc[:,dim] <= dim_boundry_cut,:]
        lhs_out, lhs_boundaries = Mondrian(lhs, quasi_identifiers, k, dim_choice_fcn, cut_choice_fcn)
        
        # The right hand side cut
        rhs = table_in.loc[table_in.loc[:,dim] > dim_boundry_cut,:]
        rhs_out, rhs_boundaries = Mondrian(rhs, quasi_identifiers, k, dim_choice_fcn, cut_choice_fcn)
        
        # Combine the partitions
        table_out = pd.concat([lhs_out, rhs_out])
        
        # Combine the boundaries definitions
        partition_boundaries = pd.concat([lhs_boundaries, rhs_boundaries])
             
    # Sorting is unnecessary in practice, but doing it so that the results are more easily examinable
    table_out = table_out.sort_values(by=quasi_identifiers).reset_index(drop=True)
    partition_boundaries = partition_boundaries.sort_values(by=quasi_identifiers).reset_index(drop=True)
    return table_out, partition_boundaries


def Mondrian_choose_cut_first_split(data,k):
    """
    A cut value choice function choosing the smallest value that permits both partitions to be size at least k.
    
    Parameters
    ----------
    data : list
        A list containing the values of the chosen dimension from the current partition. This is not a DataFrame,
        this is a list of the values from the chosen column only. That is, in your implementation you may assume that the format
        of this argument is a list of values taken from the column chosen by the dimension choice function.
        E.g., [94705, 94708, 94720, 94708, 94705].
    k : int
        The desired k value.
    
    Returns
    -------
    dimension : string
        The name of the quasi-identifier (or dimension) for the chosen cut. Otherwise, numpy.nan
    """
    if len(data) < 2 * k:
        return np.nan
    sorted_data = np.sort(data)
    
    # Get the minimum value that results in k items to the left
    cut_value = sorted_data[k - 1]
    
    # Pass i forward to find first item in list with different value
    i = k
    while i < len(sorted_data):
        if sorted_data[i] != cut_value:
            break
        i += 1
    if len(sorted_data[i:]) < k:
        return np.nan
    return cut_value
    
    # This approach is more efficient but assumes attributes are numeric
    # rhs_size = sum(data > cut_value)
    # if (rhs_size < k):
    #     return np.nan
    # return cut_value


def Mondrian_choose_dim_highest_distinct(table_partition, allowable_dims):
    """A dimension choice function choosing the dimension (with allowable cut) that has the highest number of distinct values for the next cut.
    
    Parameters
    ----------
    partition : pandas.DataFrame
        The partition on which to determine dimension for the next cut.
    allowable_dims : list
        A list containing the set of quasi-identifiers (or dimensions) with allowable cuts.
        You may assume that the provided Mondrian implementation will provide this parameter to your function correctly.
    
    Returns
    -------
    dimension : string
        The name of the quasi-identifier (or dimension) for the chosen cut.
    """
    # Count distinct values for each allowable dimension to cut on    
    distinct_counts = []
    for dim in allowable_dims:
        dim_data = table_partition[dim]
        num_distinct = len(set(dim_data))
        distinct_counts.append(num_distinct)
    # Take highest count and return it
    best_dim_index = np.argmax(distinct_counts)
    best_dimension = allowable_dims[best_dim_index]
    return best_dimension


def discernability_cost(ecs):
  k = 'count'
  sqrs = ecs[k] ** 2
  return np.sum(sqrs)


if __name__ == "__main__":
    # Sample dataset
    pass 

