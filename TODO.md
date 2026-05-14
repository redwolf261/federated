# FLEX-Persona Single-Page Frontend (Research Frontend) — Work Tracker

## Status
- ✅ Repo understood (core FL framework + existing frontend)
- ✅ Started frontend alignment with spec
- ✅ Updated animated canvas to match clustering/orbit + packet/trail + dashed centroid→server + server pulse + node styling closer to spec

## Remaining (next edits)
1. **Hero section**
   - Replace headline with exact:  
     `FLEX-Persona: Cross-Architecture Personalized Federated Learning for Non-IID and Heterogeneous Clients`
   - Ensure accent-colored word(s) match the spec intent (orange accent for key words).
   - Ensure CTA buttons are exactly as requested/approved.

2. **Metrics row**
   - Update the 4 metric cards to exact labels:
     - ↑12% Worst-Client Acc.
     - ↑8% Personalization Gain
     - ↓40% Comm. Overhead
     - 3× Faster Convergence
   - Confirm each card has a **colored top border** and correct typography.

3. **Architecture section**
   - Ensure the 5-step numbered flow exactly matches:
     1) Local train  
     2) Upload repr.  
     3) Cluster  
     4) Aggregate  
     5) Fine-tune
   - Update tech stack table to include exactly:
     - Python
     - PyTorch
     - Flower
     - scikit-learn
     - W&B
     - FEMNIST/CIFAR-100
   - Confirm privacy callout box matches brand styling and wording.

4. **Features grid**
   - Update/rename the 6 feature cards exactly to:
     - Local Client Training
     - Representation Adapter
     - Secure Repr. Sharing
     - Dynamic Clustering
     - Cluster-Level Aggregation
     - Personalized Head Fine-Tuning

5. **Evaluation panel**
   - Ensure content matches the spec:
     - Datasets: FEMNIST; CIFAR-100 Dirichlet α=0.1/0.5
     - Architectures list (confirm exact items you want)
     - Baselines: FedAvg, FedProx, IFCA, pFedMe, Ditto
     - Metrics list with colored dots (confirm exact metric texts)

6. **Footer**
   - Ensure logo + project name + tech badges match the spec.

7. **Final verification**
   - Scroll/visually verify all sections once.
   - Confirm canvas animation still performs smoothly.

## Notes
- Canvas work done in `flex-persona-frontend.jsx` (NetworkCanvas replacement).
- `frontend_app.py` (Streamlit control desk) is intentionally left unchanged unless requested.
