
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


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




if __name__ == "__main__":
    # Sample dataset
    

    # Load and print the whole table
    df = pd.read_csv('../HW3.csv')
    print(df.to_markdown())
    


    print("-"*20)
    print("3.6, k-for Sex")
    print("-"*20)
    quasi_identifiers = ["Sex"]

    
    ## get all the eqivalence classes
    k_anon, grouped_k = k_anonymity(df, quasi_identifiers)
    
    print(f"k-anonymity for {quasi_identifiers} is {k_anon}")
    print(grouped_k.to_markdown())
    


    print("-"*20)
    print("3.8, k-for Sex,Age")
    print("-"*20)
    quasi_identifiers = ["Sex", "Age"]

    
    ## get all the eqivalence classes
    k_anon, grouped_k = k_anonymity(df, quasi_identifiers)
    
    print(f"k-anonymity for {quasi_identifiers} is {k_anon}")
    print(grouped_k.to_markdown())
    

    print("-"*20)
    print("3.11, k-for Sex,Age,Marital Status")
    print("-"*20)
    quasi_identifiers = ["Sex", "Age", "Marital Status"]

    
    ## get all the eqivalence classes
    k_anon, grouped_k = k_anonymity(df, quasi_identifiers)
    
    print(f"k-anonymity for {quasi_identifiers} is {k_anon}")
    print(grouped_k.to_markdown())
    

    print("-"*20)
    print("3.11, k-for Sex, Age, Birth Country")
    print("-"*20)
    quasi_identifiers = ["Sex", "Age", "Birth Country"]

    
    ## get all the eqivalence classes
    k_anon, grouped_k = k_anonymity(df, quasi_identifiers)
    
    print(f"k-anonymity for {quasi_identifiers} is {k_anon}")
    print(grouped_k.to_markdown())
    




    print("-"*20)
    print("5.1, Bos's records")
    print("-"*20)
    quasi_identifiers = ["Sex", "Age", "Race"]

    demographics = ("M", 31, "Non-Hispanic White")
    
    ## get all the eqivalence classes
    grouped_ec =get_equivalence_classes(df, quasi_identifiers)
    ## get the spceific demographics we are looking for 
    idxs = grouped_ec[demographics] ## returns the indexes in the df
    ## Print the entries from the df
    print(df.loc[idxs].to_markdown())
    

    print("-"*20)
    print("5.4, Hard Drugs")
    print("-"*20)
    quasi_identifiers = ["Hard Drugs"]

    
    ## get all the eqivalence classes
    grouped_ec =get_equivalence_classes(df, quasi_identifiers)
    ## Get proportion of yes
    yes_frac = len(grouped_ec["Yes"])/len(df)
    print(f" The proportion of hard drug yes in the whole table is {yes_frac}")

    print("-"*20)
    print("5.6 - 5.8 Find Alex")
    print("-"*20)
    quasi_identifiers = ["Sex", "Age", "Birth Country"]
    demographics = ('M', 32, 'US')
    

    ## get all the eqivalence classes
    grouped_ec =get_equivalence_classes(df, quasi_identifiers)
    ## get the spceific demographics we are looking for 
    idxs = grouped_ec[demographics] ## returns the indexes in the df
    ## Print the entries from the df
    print(df.loc[idxs].to_markdown())




    print("-"*20)
    print("6.1, 6.2 Distinct")
    print("-"*20)
    quasi_identifiers = ["Sex"]
    sensitive_attribute = "Drinks/Day"

    satisfies_l, grouped_l = distinct_l_diversity(df, quasi_identifiers, sensitive_attribute)
    print(f"\nL-diversity satisfied: {satisfies_l}")
    print(grouped_l)
    
    

    ## get the eqivalence classes for Fmale and drinks/day
    quasi_identifiers = ["Sex"]

    attrib = ("F","Drinks/Day")
    
    ## get all the eqivalence classes
    grouped_ec =get_equivalence_classes(df, quasi_identifiers)
    print(grouped_ec)
    ## get the spceific demographics we are looking for 
    idxs = grouped_ec['F'] ## returns the indexes in the df
    ## Print the entries from the df
    print(df.loc[idxs].to_markdown())
    





    print("-"*20)
    print("6.3, Entropy")
    print("-"*20)
    quasi_identifiers = ["Sex"]
    sensitive_attribute = "Drinks/Day"

    satisfies_l, grouped_l = entropy_l_diversity(df, quasi_identifiers, sensitive_attribute)
    print(f"\nL-diversity satisfied: {satisfies_l}")
    print(grouped_l)

    print("-"*20)
    print("6.5 Distinct {Sex, Age}")
    print("-"*20)
    quasi_identifiers = ["Sex", "Age"]
    sensitive_attribute = "Drinks/Day"

    satisfies_l, grouped_l = distinct_l_diversity(df, quasi_identifiers, sensitive_attribute)
    print(f"\nL-diversity satisfied: {satisfies_l}")
    print(grouped_l)


    print("-"*20)
    print("6.7 Entroopy {Sex, Age}")
    print("-"*20)
    quasi_identifiers = ["Sex", "Age"]
    sensitive_attribute = "Drinks/Day"

    satisfies_l, grouped_l = entropy_l_diversity(df, quasi_identifiers, sensitive_attribute)
    print(f"\nL-diversity satisfied: {satisfies_l}")
    print(grouped_l)


    print("-"*20)
    print("6.9 Distinct {Sex, Age, Birth Country}")
    print("-"*20)
    quasi_identifiers = ["Sex", "Age", "Birth Country"]
    sensitive_attribute = "Drinks/Day"

    satisfies_l, grouped_l = distinct_l_diversity(df, quasi_identifiers, sensitive_attribute)
    print(f"\nL-diversity satisfied: {satisfies_l}")
    print(grouped_l)



    print("-"*20)
    print("6.11 Entropy {Sex, Age, Birth Country}")
    print("-"*20)
    quasi_identifiers = ["Sex", "Age", "Birth Country"]
    sensitive_attribute = "Drinks/Day"

    satisfies_l, grouped_l = entropy_l_diversity(df, quasi_identifiers, sensitive_attribute)
    print(f"\nL-diversity satisfied: {satisfies_l}")
    print(grouped_l)

    print("-"*20)
    print("6.13 Recursive(c,l) {Sex}")
    print("-"*20)
    quasi_identifiers = ["Sex"]
    sensitive_attribute = "Drinks/Day"

    satisfies_l, grouped_l = recursive_l_diversity(df, quasi_identifiers, sensitive_attribute, verbose=True)
    print(f"\nL-diversity satisfied: {satisfies_l}")
    print(grouped_l.to_markdown())


    print("-"*20)
    print("6.21 maimize Entropy, we can generalize the whole range of Quasi Identifiers")
    print("-"*20)
    
    sensitive_attribute = "Drinks/Day"
    ## get the unique number of Drinks/Day
    diversity_score = df[sensitive_attribute].nunique() 
    print(f"unique possibilities {diversity_score}")
