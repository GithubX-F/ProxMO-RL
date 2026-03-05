# 🚀 ProxMO: Proximity-Based Multi-Turn Optimization for LLM Agent Training

> **Practical credit assignment for multi-turn reinforcement learning with large language models**

<p align="center">
  <img src="docs/proxmo/overview.png" alt="ProxMO Framework" width="100%">
</p>

---

## ✨ Core Highlights

ProxMO is a **lightweight, practical framework** for multi-turn LLM agent training that addresses a fundamental challenge: **context-dependent credit assignment**.

<table>
<tr>
<td width="25%" align="center">

### 📊 **Superior Results**

- **+28.9%** on ALFWorld (1.5B)
- **+18.3%** on ALFWorld (7B)  
- **Best in 9/9 metrics** (7B)
- Beats GPT-4o & Gemini-2.5-Pro

</td>
<td width="25%" align="center">

### ⚡ **Production-Ready**

- **+1.09%** overhead only
- No extra networks needed
- Plug-and-play with GRPO
- Vectorized operations

</td>
<td width="25%" align="center">

### 🎯 **Two-Level Design**

- **Episode**: Success-rate aware
- **Step**: Proximity-based soft aggregation
- Independent + synergistic gains
- Hyperparameter robust

</td>
<td width="25%" align="center">

### 📈 **Comprehensive**

- Tested on 2 benchmarks
- 2 model scales (1.5B, 7B)
- Full ablation studies
- Extensive experiments

</td>
</tr>
</table>

---

## 🎯 Overview

ProxMO addresses a fundamental challenge in multi-turn LLM agent training: **context-dependent credit assignment**. 

### The Problem

Existing group-based policy optimization methods struggle with multi-turn tasks because:

- 📈 **Episode-level**: Identical statistical deviations carry vastly different informational values
  - A **failure at 90% success** likely reflects noise 🔴
  - A **success at 10% success** represents a breakthrough 🟢
  - Yet both receive identical gradient magnitudes!

- 🔍 **Step-level**: Exact state matching fragments semantically similar states into singletons
  - Strict matching → singleton groups, normalization undefined
  - Loose matching → equal weighting of dissimilar states
  - Can't distinguish action quality within trajectories

### The Solution

**ProxMO introduces two lightweight mechanisms**:

1. 🎚️ **Success-rate-aware modulation** (episode-level)
   - Adapts gradient intensity based on task difficulty
   - Amplifies rare successes in hard tasks (low success rate)
   - Attenuates noisy failures in easy tasks (high success rate)

2. 🔗 **Proximity-based soft aggregation** (step-level)
   - Replaces hard boundaries with continuous weighting
   - All states contribute proportionally to semantic proximity
   - Eliminates singleton degeneracy, enables robust baseline estimation

---

## 🧠 Method Overview

### Episode-Level: Success-Rate-Aware Modulation

ProxMO adapts gradient intensity to task difficulty using empirical success rate `p`:

```
Low-success regime (p → 0):  🔴 Rare successes amplified (w ≈ 1.05)
                             ✓ Consolidate breakthroughs
                             
Medium-success regime (p ≈ 0.5): Minimal modulation (w ≈ 1.0)
                                ✓ Balanced learning
                                
High-success regime (p → 1):  Noisy failures attenuated (w ≈ 0.95)
                             ✓ Reduce over-correction
```

**Mathematical formulation**:
```
w(R, p) = 1 + β · f(R, p)

where f(R, p) uses sigmoid-based functions to detect task difficulty
and adjust gradient strength asymmetrically
```

### Step-Level: Proximity-Based Soft Aggregation

Instead of hard state matching, ProxMO computes baselines via continuous similarity weighting:

```
Traditional (GiGPO):         ProxMO:
┌─────────┐                 ┌──────────┐
│ State   │ Exact Match?    │ State    │ TF-IDF Similarity
│ Cluster │ ↓ Yes/No        │ Weights  │ ↓ Continuous
└─────────┘                 └──────────┘
  Hard boundaries             Soft weighting
  Singleton degeneracy        Robust estimation
```

**Key advantages**:
- ✅ All states contribute proportionally to semantic proximity
- ✅ Eliminates singleton groups in high-dimensional spaces
- ✅ Smooth gradient flow via continuous weighting

---

## 📊 Results

ProxMO consistently outperforms existing methods on challenging multi-turn interactive benchmarks:

### Qwen2.5-1.5B-Instruct

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | **All** | Score | **Succ.** |
|--------|------|------|-------|------|------|-------|--------|-------|-----------|
| **Closed-Source Models** |
| GPT-4o | 75.3 | 60.8 | 31.2 | 56.7 | 21.6 | 49.8 | 48.0 | 31.8 | 23.7 |
| Gemini-2.5-Pro | 92.8 | 63.3 | 62.1 | 69.0 | 26.6 | 58.7 | 60.3 | 42.5 | 35.9 |
| **Open-Source Baselines** |
| Base | 5.9 | 5.5 | 3.3 | 9.7 | 4.2 | 0.0 | 4.1 | 25.1 | 6.3 |
| ReAct | 17.4 | 20.5 | 15.7 | 6.2 | 7.7 | 2.0 | 12.8 | 42.1 | 14.3 |
| Reflexion | 37.8 | 24.0 | 23.3 | 14.5 | 20.7 | 3.9 | 23.5 | 58.6 | 23.5 |
| **RL Methods** |
| GRPO | 80.0 | 50.0 | 75.0 | 88.9 | 63.2 | 50.0 | 70.3 | 73.1 | 52.2 |
| GiGPO | **95.3** | 80.2 | **92.9** | **92.7** | 70.6 | 78.5 | 85.2 | 81.7 | 62.3 |
| **ProxMO (Ours)** | 94.3 | **92.9** ✨ | 89.3 | 92.2 | **89.5** ✨ | **87.0** ✨ | **90.6** ✨ | **85.3** ✨ | **67.1** ✨ |
| Δ vs GRPO | +17.9% | +85.8% | +19.1% | +3.7% | +41.7% | +74.0% | **+28.9%** | **+16.7%** | **+28.6%** |

### Qwen2.5-7B-Instruct

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | **All** | Score | **Succ.** |
|--------|------|------|-------|------|------|-------|--------|-------|-----------|
| **Closed-Source Models** |
| GPT-4o | 75.3 | 60.8 | 31.2 | 56.7 | 21.6 | 49.8 | 48.0 | 31.8 | 23.7 |
| Gemini-2.5-Pro | 92.8 | 63.3 | 62.1 | 69.0 | 26.6 | 58.7 | 60.3 | 42.5 | 35.9 |
| **Open-Source Baselines** |
| Base | 34.8 | 22.9 | 18.1 | 7.3 | 2.5 | 3.6 | 16.2 | 25.1 | 8.4 |
| ReAct | 50.1 | 33.8 | 35.7 | 12.5 | 17.3 | 18.9 | 29.8 | 47.8 | 21.0 |
| Reflexion | 63.4 | 40.2 | 46.5 | 29.7 | 37.9 | 22.6 | 44.1 | 56.3 | 30.2 |
| **RL Methods** |
| GRPO | 90.7 | 66.2 | 94.1 | 91.2 | 78.9 | 70.5 | 79.8 | 79.2 | 67.2 |
| GiGPO | 97.5 | 81.3 | 88.5 | 85.7 | 90.0 | 83.5 | 89.5 | 85.5 | 74.8 |
| **ProxMO (Ours)** | **98.4** ✨ | **88.6** ✨ | **95.7** ✨ | **93.8** ✨ | **91.3** ✨ | **89.8** ✨ | **94.5** ✨ | **87.2** ✨ | **76.5** ✨ |
| Δ vs GRPO | +8.5% | +33.8% | +1.7% | +2.8% | +15.6% | +27.5% | **+18.3%** | **+10.1%** | **+13.8%** |

### 🎯 Key Insights

<table>
<tr>
<td>

#### ✅ **Best-in-Class Performance**
- **9/9 metrics** best on Qwen2.5-7B
- **Wins** most categories on 1.5B
- Beats both closed-source LLMs on many tasks

#### 🚀 **Exceptional Gains on Hard Tasks**
- Look: **+85.8%** (1.5B), **+33.8%** (7B) 
- Cool: **+41.7%** (1.5B), **+15.6%** (7B)
- Pick2: **+74.0%** (1.5B), **+27.5%** (7B)
- These are long-horizon tasks → prove step-level credit assignment works

</td>
<td>

#### 🔄 **Consistent Across Settings**
- Both model scales (1.5B & 7B)
- Both benchmarks (ALFWorld & WebShop)  
- All task categories
- No regression on any metric

#### ⚡ **Production Deployment Ready**
- Only **+1.09%** overhead vs GRPO
- **No extra networks** (unlike critic-based methods)
- **Plug-and-play** integration
- **Vectorized operations** for efficiency

</td>
</tr>
</table>

---

### 📈 Performance Comparison Summary

| Metric | GRPO → ProxMO (1.5B) | GRPO → ProxMO (7B) | Win Rate |
|--------|-------------------|------------------|----------|
| **ALFWorld Overall** | 70.3 → **90.6** ✨ | 79.8 → **94.5** ✨ | **2/2** |
| **WebShop Score** | 73.1 → **85.3** ✨ | 79.2 → **87.2** ✨ | **2/2** |
| **WebShop Success** | 52.2 → **67.1** ✨ | 67.2 → **76.5** ✨ | **2/2** |
| **vs GiGPO** | **90.6** vs 85.2 | **94.5** vs 89.5 | **2/2** |

🏆 **Absolute Winner**: ProxMO beats both GRPO baseline and GiGPO across all major metrics

## Installation

### 🐍 Setup Environment

```bash
conda create -n proxmo python=3.10 -y
conda activate proxmo
pip install -e .
```

### 🌍 Install ALFWorld Environment

**📦 Install dependencies**:
```bash
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
```

**⬇️ Download required assets** (stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

**✅ Test the installation**:
```bash
alfworld-play-tw
```

### 🛒 Install WebShop Environment

> ⚠️ **Important**: WebShop requires Python ≤3.10. Create a separate environment if needed.

**📥 Install WebShop**:
```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

**💡 Note**: If you encounter issues with Google Drive downloads, manually download the required files or use alternative download methods.

## Quick Start

After installing the environments and dependencies, run ProxMO training with:

**🎮 Train on ALFWorld**:
```bash
bash examples/proxmo_trainer/run_alfworld.sh
```

**🛍️ Train on WebShop**:
```bash
bash examples/proxmo_trainer/run_webshop.sh
```

These scripts handle all configuration and setup automatically. Refer to the scripts for customization options.

## Code Structure

```
├── verl/
│   ├── trainer/
│   │   ├── main_ppo.py          # Training entry point
│   │   └── ppo/
│   │       └── ray_trainer.py    # Ray-distributed training
│   └── workers/
│       └── reward_manager/
│           └── proxmo.py         # ProxMO algorithm implementation
├── agent_system/
│   └── environments/
│       ├── env_package/
│       │   ├── alfworld/         # ALFWorld environment
│       │   └── webshop/          # WebShop environment
│       └── env_manager.py        # Environment management
└── examples/
    └── proxmo_trainer/           # Training scripts
        ├── run_alfworld.sh
        └── run_webshop.sh
```

## 🎁 Key Features

### 🔄 **Multi-Turn Agent Training**
- ✅ **Step-wise interaction loops** with environment feedback integration
- ✅ **Customizable memory modules** for history management and context tracking
- ✅ **Flexible per-step input structures** supporting diverse observation types
- 📌 Works with any sequential decision-making task (embodied, web, text-based)

### ⚡ **Scalable and Efficient**  
- ✅ **Parallelized environment rollouts** (no speed degradation)
- ✅ **Efficient group-based advantage estimation** (no critic networks)
- ✅ **Minimal computational overhead** (+1.09% vs GRPO)
- 📊 Vectorized TF-IDF computation, O(N²) complexity fully parallelizable

### 🚀 **Production-Ready**
- ✅ **Plug-and-play integration** with existing GRPO pipelines
- ✅ **Hyperparameter-robust design** (stable across domains & scales)
- ✅ **No architectural changes** required to existing models
- 🛡️ Drop-in replacement with immediate deployment capability

### 📈 **Comprehensive Validation**
- ✅ **Extensive experiments** on 2 benchmarks (ALFWorld, WebShop)
- ✅ **Multiple model scales** tested (1.5B, 7B parameters)
- ✅ **Full ablation studies** proving independent mechanism contributions
- ✅ **Hyperparameter sensitivity analysis** demonstrating robustness
- 📋 Consistency across diverse task categories and difficulty regimes
