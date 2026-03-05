# Copyright 2025 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based ProxMO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of ProxMO ----------------- #
# ---------------------------------------------------------- #

def compute_proxmo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   return_components: bool = False,
                                   enable_psc: bool = False,
                                   psc_alpha: float = 5.0,
                                   psc_beta: float = 0.866,
                                   ):

    if mode == "mean_std_norm": # 标准 GRPO 归一化
        remove_std = False
        soft_grouping = False 
    elif mode == "mean_norm": # ProxMO w/o std
        remove_std = True
        soft_grouping = False
    elif mode == "soft_grouping":
        remove_std = True
        soft_grouping = True
        if TfidfVectorizer is None:
             raise ImportError("scikit-learn is required for soft_grouping mode. Please install it via `pip install scikit-learn`.")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std, 
                                             enable_psc=enable_psc, psc_alpha=psc_alpha, psc_beta=psc_beta)
    
    if soft_grouping:
         step_advantages = step_soft_reward(step_rewards, response_mask, anchor_obs, index, epsilon=epsilon)
    else:
        # Anchor state grouping (Eq. 6 in the paper).
        step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

        # Compute step relative advantages (Eq. 7 in the paper).
        step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    if return_components:
        return episode_advantages, step_advantages

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores

def compute_psc_weights(rewards: torch.Tensor, p: float, alpha: float, beta: float):
    """
    Compute weights for Polarized Signal Controller (PSC).
    Args:
        rewards: (bs,)
        p: float, success rate of the group
        alpha: float, steepness of the sigmoid function
        beta: float, scaling factor
    Returns:
        weights: (bs,)
    """
    d = 1.0 - p
    
    # Success (reward=1): w = 1 + beta * (sigmoid(d^alpha) - 0.5)
    # Failure (reward=0): w = 1 + beta * (sigmoid(-p^alpha) - 0.5)
    
    # f_success = sigma(d^alpha) - 0.5
    f_success = torch.sigmoid(torch.tensor(d ** alpha)) - 0.5
    w_pos = 1.0 + beta * f_success
    
    # f_failure = sigma(-p^alpha) - 0.5
    f_failure = torch.sigmoid(torch.tensor(-(p ** alpha))) - 0.5
    w_neg = 1.0 + beta * f_failure
    
    # Apply weights based on rewards
    # Assume rewards are roughly 0 or 1.
    weights = torch.where(rewards > 0.5, w_pos, w_neg)
    return weights

def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        enable_psc: bool = False,
                        psc_alpha: float = 5.0,
                        psc_beta: float = 0.866,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for ProxMO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
        enable_psc: bool
            If True, apply Polarized Signal Controller (PSC) weighting.
        psc_alpha: float
            Steepness of the sigmoid function for PSC.
        psc_beta: float
            Scaling factor for PSC.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            # Store original score for PSC calculation if needed
            original_score = scores[i].clone() if enable_psc else None
            
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            
            # Apply PSC weighting
            if enable_psc:
                p = id2mean[index[i]].item()
                # Ensure p is within [0, 1] just in case
                p = max(0.0, min(1.0, p))
                
                # Note: original_score should be the reward (0 or 1)
                # We assume the task rewards are 0/1.
                w = compute_psc_weights(original_score, p, psc_alpha, psc_beta)
                scores[i] = scores[i] * w
                
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages

def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids

def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for ProxMO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages

def step_soft_reward(step_rewards: torch.Tensor,
                     response_mask: torch.Tensor,
                     anchor_obs: np.array,
                     index: np.array,
                     epsilon: float = 1e-6,
                     temperature: float = 0.1, # Temperature for softmax
                     ):
    """
    Compute step-level advantage using soft grouping (TF-IDF + GPU Matrix Operation).
    
    Args:
        step_rewards: (bs,)
        response_mask: (bs, response_length)
        anchor_obs: (bs,) - text observations
        index: (bs,) - episode_uid (prompt_uid)
        
    Returns:
        step_advantages: (bs, response_length)
    """
    bsz = step_rewards.shape[0]
    device = step_rewards.device
    response_length = response_mask.shape[-1]
    
    # Pre-compute TF-IDF for ALL observations to avoid re-fitting vectorizer many times
    try:
        vectorizer = TfidfVectorizer(max_features=1024) # Limit features to keep it fast
        # Convert all observations to string to be safe
        anchor_obs_str = [str(o) for o in anchor_obs]
        tfidf_matrix_all = vectorizer.fit_transform(anchor_obs_str) # Sparse matrix (bs, vocab)
        tfidf_tensor_all = torch.tensor(tfidf_matrix_all.toarray(), dtype=torch.float32, device=device)
    except ValueError:
         # Handle case where vocabulary is empty or other sklearn errors
         # Fallback to zeros
         return torch.zeros((bsz, response_length), device=device) * response_mask

    # Normalize vectors globally (L2 norm)
    # (bs, vocab)
    norm = tfidf_tensor_all.norm(dim=1, keepdim=True)
    tfidf_tensor_all = tfidf_tensor_all / (norm + 1e-8)
    
    # Compute Global Similarity Matrix (Cosine Similarity)
    # (bs, bs)
    # Since vectors are normalized, dot product is cosine similarity
    sim_matrix = tfidf_tensor_all @ tfidf_tensor_all.T
    
    # Generate Group Mask
    # We need to map 'index' (which might be strings) to integers for tensor operations
    # np.unique returns sorted unique elements and the indices to reconstruct the original array
    _, group_indices = np.unique(index, return_inverse=True)
    group_indices_tensor = torch.tensor(group_indices, device=device)
    
    # mask[i, j] is True if i and j are in the same group
    # (bs, 1) == (1, bs) -> (bs, bs)
    group_mask = group_indices_tensor.unsqueeze(1) == group_indices_tensor.unsqueeze(0)
    
    # Apply Temperature and Mask
    # We set non-group entries to -inf so they don't contribute to softmax
    masked_sim = sim_matrix / temperature
    masked_sim = masked_sim.masked_fill(~group_mask, float('-inf'))
    
    # Compute Soft Weights
    # (bs, bs)
    weights = torch.softmax(masked_sim, dim=1)
    
    # Calculate Baseline (Expected Reward for this state)
    # baseline[i] = sum_j (weights[i,j] * rewards[j])
    # (bs, bs) @ (bs,) -> (bs,)
    baseline = weights @ step_rewards
    
    # Compute Advantage
    soft_advantages = step_rewards - baseline

    # Expand to response length
    step_advantages = soft_advantages.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages
