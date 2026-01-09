"""
Recommendation system functions for Instacart project
All recommendation approaches: Baseline, CF, CBF, Hybrid
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ================================================================
# BASELINE RECOMMENDATIONS
# ================================================================

def get_baseline_recommendations(user_id, global_popularity, user_purchase_cache, n=10):
    """Global baseline: most popular products"""
    user_purchases = user_purchase_cache.get(user_id, set())
    recommendations = global_popularity[~global_popularity['product_id'].isin(user_purchases)]
    return list(recommendations['product_id'].head(n))


def get_segment_baseline_recommendations(user_id, cluster_id, segment_popularity, 
                                        user_purchase_cache, n=10):
    """Segment-specific baseline: popular products within segment"""
    user_purchases = user_purchase_cache.get(user_id, set())
    segment_pop = segment_popularity[cluster_id]
    recommendations = segment_pop[~segment_pop['product_id'].isin(user_purchases)]
    return list(recommendations['product_id'].head(n))


# ================================================================
# COLLABORATIVE FILTERING
# ================================================================

def get_cf_recommendations(model, user_id, all_products, user_purchase_cache, 
                          n=10, exclude_purchased=True):
    """
    CF recommendations using trained SVD model
    Works for both global and segment-specific models
    """
    if exclude_purchased:
        purchased = user_purchase_cache.get(user_id, set())
        candidate_products = [p for p in all_products if p not in purchased]
    else:
        candidate_products = all_products
    
    predictions = []
    for product_id in candidate_products:
        pred = model.predict(user_id, product_id)
        predictions.append((product_id, pred.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


# ================================================================
# CONTENT-BASED FILTERING
# ================================================================

def create_user_profile(user_id, purchase_counts, item_profile):
    """Create weighted user profile based on purchase history"""
    user_purchases = purchase_counts[purchase_counts['user_id'] == user_id]
    
    if len(user_purchases) == 0:
        return None
    
    purchased_items = user_purchases['product_id'].values
    purchase_freqs = user_purchases['frequency'].values
    
    log_freqs = np.log1p(purchase_freqs)
    weights = log_freqs / log_freqs.sum()
    
    item_vectors = item_profile.loc[
        item_profile['product_id'].isin(purchased_items)
    ].drop('product_id', axis=1).values
    
    user_profile = (item_vectors.T @ weights).reshape(1, -1)
    return user_profile


def get_cbf_recommendations(user_id, purchase_counts, item_profile, n=10):
    """Content-based recommendations using item features"""
    user_profile = create_user_profile(user_id, purchase_counts, item_profile)
    
    if user_profile is None:
        return []
    
    user_products = purchase_counts[purchase_counts['user_id'] == user_id]['product_id'].values
    
    feature_matrix = item_profile.drop('product_id', axis=1).values
    similarities = cosine_similarity(user_profile, feature_matrix)[0]
    
    idx_to_product = {idx: pid for idx, pid in enumerate(item_profile['product_id'])}
    
    recommendations = []
    for idx, score in enumerate(similarities):
        product_id = idx_to_product[idx]
        if product_id not in user_products:
            recommendations.append((product_id, score))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]


# ================================================================
# HYBRID RECOMMENDATIONS
# ================================================================

def get_hybrid_recommendations(cf_model, user_id, all_products, user_purchase_cache,
                               purchase_counts, item_profile, n=10, 
                               cf_weight=0.5, cbf_weight=0.5):
    """
    Hybrid: CF + CBF with weighted combination
    Works for both global and segment-specific CF models
    """
    # Get CF recommendations
    cf_recs = get_cf_recommendations(cf_model, user_id, all_products, 
                                     user_purchase_cache, n=400, exclude_purchased=True)
    
    # Get CBF recommendations
    cbf_recs = get_cbf_recommendations(user_id, purchase_counts, item_profile, n=400)
    
    # Normalize CF scores
    if len(cf_recs) > 0:
        cf_scores_dict = {pid: score for pid, score in cf_recs}
        cf_min, cf_max = min(cf_scores_dict.values()), max(cf_scores_dict.values())
        cf_range = cf_max - cf_min if cf_max > cf_min else 1
        cf_scores_norm = {pid: (score - cf_min) / cf_range for pid, score in cf_scores_dict.items()}
    else:
        cf_scores_norm = {}
    
    # Normalize CBF scores
    if len(cbf_recs) > 0:
        cbf_scores_dict = {pid: score for pid, score in cbf_recs}
        cbf_min, cbf_max = min(cbf_scores_dict.values()), max(cbf_scores_dict.values())
        cbf_range = cbf_max - cbf_min if cbf_max > cbf_min else 1
        cbf_scores_norm = {pid: (score - cbf_min) / cbf_range for pid, score in cbf_scores_dict.items()}
    else:
        cbf_scores_norm = {}
    
    # Combine
    all_prods = set(cf_scores_norm.keys()) | set(cbf_scores_norm.keys())
    hybrid_scores = {
        pid: (cf_weight * cf_scores_norm.get(pid, 0)) + (cbf_weight * cbf_scores_norm.get(pid, 0))
        for pid in all_prods
    }
    
    return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n]


# ================================================================
# EVALUATION METRICS
# ================================================================

def precision_at_k(recommended, actual, k):
    """Precision@K"""
    rec_k = recommended[:k]
    relevant = set(rec_k) & set(actual)
    return len(relevant) / k if k > 0 else 0


def recall_at_k(recommended, actual, k):
    """Recall@K"""
    rec_k = recommended[:k]
    relevant = set(rec_k) & set(actual)
    return len(relevant) / len(actual) if len(actual) > 0 else 0


def f1_at_k(recommended, actual, k):
    """F1@K"""
    prec = precision_at_k(recommended, actual, k)
    rec = recall_at_k(recommended, actual, k)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0