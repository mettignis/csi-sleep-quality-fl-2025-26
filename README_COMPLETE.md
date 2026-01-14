# 🏆 Sleep Quality Prediction under Federated Learning Constraints

**Kaggle Competition Result: 3rd Place 🥉**  
**Final Score:** 11.79207 MAE  
**Competition Period:** December 2025 - January 9, 2026

**Authors:** Anna Mettifogo, Gabriel Fumagalli, Niccolò Kadera  
**Course:** Creativity, Science and Innovation - Politecnico di Milano  
**Project:** Behavioral and Ambient Learning Across Networked Collaborative Edges (BALANCE)

---

## 📖 Abstract

This work investigates sleep quality prediction from wearable-device biometric data under strict federated learning constraints, where data privacy prohibits centralized data aggregation. The challenge is characterized by an extremely small dataset (999 training samples distributed across nine federated clients), leading to significant overfitting in baseline models. 

To address this, we propose a federated approach combining **per-client data augmentation** and **heterogeneous ensemble learning**. An ensemble of Random Forest and LightGBM models was trained on augmented data using 40 engineered statistical features extracted from time-series biometrics (heart rate, respiration, stress). 

**Key Finding:** In small federated datasets, mitigating overfitting through data augmentation (6x expansion) and architectural diversity is substantially more effective than extensive hyperparameter optimization.

**Final Result:** Through systematic experimentation (42 experiments, 18 submissions), our meta-ensemble strategy combining augmented training (Exp 039) with multi-algorithm diversity (Exp 031) achieved **11.79207 MAE**, securing 3rd place and demonstrating that data quality and ensemble robustness matter more than model complexity in privacy-constrained settings.

---

## 🎯 Problem Statement & Constraints

### Federated Learning Requirements

The competition enforces a **strict Federated Learning paradigm** with the following constraints:

- ❌ **NO centralized data aggregation**: Raw data must remain private to each client
- ❌ **NO label sharing**: Target labels cannot be aggregated across clients
- ✅ **Model/Prediction sharing**: Only model weights and predictions can be exchanged
- ✅ **Per-client augmentation**: Data synthesis allowed locally without sharing

**Evaluation:** Mean Absolute Error (MAE) computed on a centralized test set of 189 samples.

### Dataset Characteristics

**Data Collection:** Self-collected biometric data from 45 participants wearing Garmin Vivoactive 5 devices over one month, monitoring sleep and daily activity patterns.

**Distribution:**
- **999 training samples** distributed across 9 federated clients (~110 samples/client)
- **189 test samples** (shared, unlabeled)
- **45 users** organized in 9 groups (5 users per client)

**Features:** 
- **27 raw features** → expanded to **40 engineered features**
- Time-series biometrics: `hr_time_series`, `resp_time_series`, `stress_time_series`
- Daily aggregates: resting HR, min/max HR, respiration during sleep/wake, stress levels
- **Special values:** -1 (missing), -2 (invalid), null (sensor gaps)

**Challenges:**
- ⚠️ **Extreme data scarcity**: ~110 samples/client → severe overfitting risk
- ⚠️ **High noise**: Wearable sensor variability and measurement errors
- ⚠️ **Client heterogeneity**: Different physiological baselines across users
- ⚠️ **Validation instability**: Small validation sets (~27 samples/client) → unreliable metrics

---

[Continue with rest of the comprehensive README content...]

## 📚 References

1. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 2017*.
2. Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.
3. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
4. McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS 2017*.

---

## 📄 License

Academic project for Politecnico di Milano - CSI Course (2025-2026)

---

## 🙏 Acknowledgments

We thank Professor Marco D. Santambrogio and Aldo Torrebruno for their guidance throughout this project. Special thanks to all course participants who contributed their biometric data to create this unique federated learning dataset.

**Competition Stats:**
- 42 experiments documented
- 18 Kaggle submissions (out of 35 allowed)
- 5 breakthrough moments
- 2 weeks of intense iteration
- 🥉 3rd place finish

**Key Takeaway:** In machine learning competitions with strict privacy constraints, data quality/quantity and ensemble diversity matter more than model complexity. Our winning strategy prioritized per-client data augmentation (6x samples) and architectural heterogeneity over hyperparameter tuning, demonstrating that a data-centric approach is essential in small federated datasets.
