# Ransomware Detection Research Proposal

## Problem Definition (as defined by Dr. Fujinoki)
Detect or predict ransomware activity based on the way data is updated (e.g., encrypted, obfuscated, shuffled, network communication to C2 servers) with malicious intent. By analyzing the requests to update the contents in production data, we aim to develop a detection system that can identify, predict, and explain potential ransomware activity before it causes significant damage.

## Objective
To develop a system that addresses Dr. Fujinoki's problem definition by proactively detecting or predicting ransomware activity through analysis of data update patterns, focusing on identification before significant damage occurs.

## Research Hypothesis
A sequence prediction model leveraging comprehensive data from file system operations, process activities, and network traffic can effectively detect a wide range of ransomware activities, learning subtle patterns of malicious behavior and distinguishing them from normal operations.

## Challenges
1. Fine-grained monitoring of data update requests
2. Analyzing complex, high-volume data from multiple sources
3. Achieving proactive, near real-time detection
4. Minimizing false negatives while maintaining a tolerable false positive rate
5. Distinguishing between legitimate and malicious data updates (e.g., encryption, obfuscation, shuffling)
6. Detecting sparse ransomware activity in large volumes of benign data

## Proposed Approach: Multi-Source Sequence Prediction

### Data Collection and Annotation
- **Sources:** 
  - File system operations (Auditd)
  - Process activities (Strace)
  - Network traffic (Suricata)
- **Types:** 
  - Real-world ransomware activity datasets
  - Synthetic data from controlled experiments, including various update patterns (encryption, obfuscation, shuffling)
- **Annotation:**
  - Synthetic data will be augmented with detailed annotations describing specific ransomware activities
  - Example annotation: "[potential ransomware activity] C2 communication to exchange encryption keys"
  - These annotations will serve as explanations for the model to learn and predict
- **Controlled Sparsity:**
  - Synthetic datasets will be generated with varying levels of ransomware activity sparsity
  - Sparsity levels will range from dense (easy to detect) to extremely sparse (challenging to detect)
  - This controlled sparsity allows for comprehensive evaluation of the model's performance under different conditions

### Model
- **Type:** Sequence prediction model (e.g., Transformer)
- **Input:** Multi-source data sequences, including patterns of data updates and network communications
- **Output:** 
  - Predictions of potential ransomware activity
  - Explanations based on learned annotations from synthetic data

### Training Strategy
1. Train the model on annotated synthetic data with varying levels of ransomware activity sparsity
2. Fine-tune on real-world data to improve generalization
3. Validate the model's ability to provide accurate predictions and explanations on unseen data, particularly focusing on sparse activity scenarios

### Experimental Setup
1. Controlled environment with unrestricted computational resources
2. Comprehensive data collection from multiple tools
3. Synthetic ransomware activity generation (e.g., using WannaLaugh emulator) with detailed annotations and controlled sparsity
4. Model training on combined dataset (synthetic and real)
5. Evaluation on out-of-distribution data, including extremely sparse ransomware activity scenarios
6. Iterative refinement of model, dataset, annotation strategy, and sparsity levels

## Expected Outcomes
1. A robust ransomware detection system with high accuracy and low false negatives, capable of identifying various malicious data update patterns
2. A model that can provide human-readable explanations for its predictions, leveraging learned annotations
3. Insights into effective multi-source data integration and annotation strategies for cybersecurity
4. Understanding of model performance across different levels of ransomware activity sparsity
5. Potential for adaptation to real-time production environments (future work)

## References
1. WannaLaugh: A Configurable Ransomware Emulator. arXiv:2403.07540

