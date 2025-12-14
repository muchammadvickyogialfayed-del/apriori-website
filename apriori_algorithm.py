"""
Implementasi Algoritma Apriori untuk Market Basket Analysis
Author: Data Mining Project
"""

from itertools import combinations
from collections import defaultdict
import pandas as pd


class AprioriAlgorithm:
    def __init__(self, min_support=0.2, min_confidence=0.5):
        """
        Initialize Apriori Algorithm
        
        Parameters:
        -----------
        min_support : float
            Minimum support threshold (0-1)
        min_confidence : float
            Minimum confidence threshold (0-1)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.frequent_itemsets = {}
        self.association_rules = []
        
    def load_transactions(self, transactions_list):
        """
        Load transactions data
        
        Parameters:
        -----------
        transactions_list : list of lists
            Each transaction is a list of items
        """
        self.transactions = [set(transaction) for transaction in transactions_list]
        
    def calculate_support(self, itemset):
        """
        Calculate support for an itemset
        
        Parameters:
        -----------
        itemset : set
            Set of items
            
        Returns:
        --------
        float : support value
        """
        count = sum(1 for transaction in self.transactions if itemset.issubset(transaction))
        return count / len(self.transactions)
    
    def get_items(self):
        """
        Get all unique items from transactions
        
        Returns:
        --------
        set : all unique items
        """
        items = set()
        for transaction in self.transactions:
            items.update(transaction)
        return items
    
    def generate_candidates(self, itemsets, k):
        """
        Generate candidate itemsets of size k
        
        Parameters:
        -----------
        itemsets : list of sets
            Frequent itemsets of size k-1
        k : int
            Size of new itemsets to generate
            
        Returns:
        --------
        list of sets : candidate itemsets
        """
        candidates = []
        n = len(itemsets)
        
        for i in range(n):
            for j in range(i + 1, n):
                union = itemsets[i] | itemsets[j]
                if len(union) == k:
                    candidates.append(union)
        
        # Remove duplicates
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
                
        return unique_candidates
    
    def find_frequent_itemsets(self):
        """
        Find all frequent itemsets using Apriori algorithm
        
        Returns:
        --------
        dict : {itemset_size: [(itemset, support), ...]}
        """
        # Get all unique items
        items = self.get_items()
        
        # Find frequent 1-itemsets
        frequent_1 = []
        for item in items:
            itemset = frozenset([item])
            support = self.calculate_support(itemset)
            if support >= self.min_support:
                frequent_1.append(itemset)
        
        self.frequent_itemsets[1] = [(itemset, self.calculate_support(itemset)) 
                                      for itemset in frequent_1]
        
        # Find frequent k-itemsets
        k = 2
        current_frequent = frequent_1
        
        while current_frequent:
            # Generate candidates
            candidates = self.generate_candidates(current_frequent, k)
            
            # Filter by minimum support
            frequent_k = []
            for candidate in candidates:
                support = self.calculate_support(candidate)
                if support >= self.min_support:
                    frequent_k.append(candidate)
            
            if frequent_k:
                self.frequent_itemsets[k] = [(itemset, self.calculate_support(itemset)) 
                                              for itemset in frequent_k]
                current_frequent = frequent_k
                k += 1
            else:
                break
        
        return self.frequent_itemsets
    
    def generate_association_rules(self):
        """
        Generate association rules from frequent itemsets
        
        Returns:
        --------
        list : [(antecedent, consequent, support, confidence, lift), ...]
        """
        self.association_rules = []
        
        # Generate rules from itemsets of size 2 or more
        for size in range(2, max(self.frequent_itemsets.keys()) + 1):
            if size not in self.frequent_itemsets:
                continue
                
            for itemset, support in self.frequent_itemsets[size]:
                # Generate all possible rules
                items = list(itemset)
                
                # Try all possible splits
                for i in range(1, len(items)):
                    for antecedent in combinations(items, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        # Calculate confidence
                        antecedent_support = self.calculate_support(antecedent)
                        if antecedent_support > 0:
                            confidence = support / antecedent_support
                            
                            if confidence >= self.min_confidence:
                                # Calculate lift
                                consequent_support = self.calculate_support(consequent)
                                lift = confidence / consequent_support if consequent_support > 0 else 0
                                
                                self.association_rules.append({
                                    'antecedent': set(antecedent),
                                    'consequent': set(consequent),
                                    'support': support,
                                    'confidence': confidence,
                                    'lift': lift
                                })
        
        # Sort by confidence
        self.association_rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        return self.association_rules
    
    def get_frequent_itemsets_df(self):
        """
        Convert frequent itemsets to DataFrame
        
        Returns:
        --------
        pandas.DataFrame
        """
        data = []
        for size, itemsets in sorted(self.frequent_itemsets.items()):
            for itemset, support in itemsets:
                data.append({
                    'Itemset': ', '.join(sorted(itemset)),
                    'Size': size,
                    'Support': support,
                    'Support (%)': f"{support * 100:.2f}%"
                })
        
        return pd.DataFrame(data)
    
    def get_association_rules_df(self):
        """
        Convert association rules to DataFrame
        
        Returns:
        --------
        pandas.DataFrame
        """
        data = []
        for rule in self.association_rules:
            data.append({
                'Antecedent (Jika)': ', '.join(sorted(rule['antecedent'])),
                'Consequent (Maka)': ', '.join(sorted(rule['consequent'])),
                'Support': f"{rule['support'] * 100:.2f}%",
                'Confidence': f"{rule['confidence'] * 100:.2f}%",
                'Lift': f"{rule['lift']:.2f}"
            })
        
        return pd.DataFrame(data)
