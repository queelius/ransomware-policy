## Introduction

### Objective

Detect or predict ransomware activity based on the way data is updated (e.g., encrypted, obfuscated, shuffled, network communication to C2 servers) with malicious intent. By analyzing the requests to update the contents in production data, we aim to develop a detection system that can identify, predict, and explain potential ransomware activity before it causes significant damage.

For instance, instead of just looking for encryption after it happens, the system might analyze the pattern of requests made to the data (e.g., sequences of file writes, access patterns) to detect suspicious activity, such as gradual data encryption (salami-slice method), obfuscation, or shuffling, which might not be immediately recognized as ransomware using traditional detection methods.

## Challenges

### Granularity of Detection

The research requires fine-grained monitoring of data update requests, which is challenging because ransomware may employ methods that disguise their true intentions (e.g., gradually manipulating the data).

### Data Abundance and Complexity

Ransomware detection requires analyzing a wide range of data sources, including file system operations, network traffic, and process activities. The volume and complexity of this data can make it challenging to identify subtle patterns of malicious activity. If the ransomware activity is "dense" in the data, detecting it is more likely. However, real data may contain very sparse ransomware activity, and the detection system must be capable of finding the needle in the haystack.

### Proactive Detection

The goal is to detect malicious activity as it happens, rather than reacting after the damage is done. This requires sophisticated approaches that can discern malicious intent from legitimate data operations in near real-time, which can be computationally expensive.

### Negligible False Negative Rate

We aim to achieve a false negative rate close to zero, meaning the system should rarely miss malicious updates. This is a challenge, especially with the evolution of ransomware techniques like the salami-slice method or obfuscation that can evade traditional detection methods.

### Tolerable False Positives

While striving for a negligible false negative rate, the system must tolerate a reasonably small false positive rate. Excessive false positives can reduce the usability of the system, leading to alarm fatigue.

## Possible Approach: Sequence Prediction from Multiple Sources

Given the sophistication of ransomware techniques, a multi-source data collection approach may be necessary to detect subtle patterns of malicious activity. A sequence prediction model that learns from multiple sources of data (e.g., file system operations, process activities, network traffic) may be appropriate.

The training data can be both real and synthetic. For synthetic data, we can perform controlled experiments and provide detailed annotations or explanations about the ransomware activity, such as:
```
[potential ransomware activity] C2 communication to exchange encryption keys
```
The sequence model will learn to associate these annotations with system calls, network traffic, etc., and predict similar annotations in a real-world system, enabling proactive detection.

**Assumption:** The system operates in an ideal research environment with no computational or resource constraints. The focus is on developing and testing detection methods without the limitations of real-time production systems.

## Tools and Setup

We have two general approaches for collecting training data:

1. **Existing real-world ransomware activity datasets:** Real data provides authenticity but may lack controlled richness.
2. **Synthetic ransomware activity generated in a controlled environment:** Synthetic data allows for control and richness but may not be as realistic as real-world data.

### Tools for Data Collection

- **Auditd:** Monitors file system operations and process activities.
- **Suricata:** Analyzes network traffic for anomalies associated with ransomware (e.g., C2 communication, unusual DNS queries).
- **Strace:** Provides detailed tracing of system calls made by processes, especially file I/O and network-related calls.
- **Ransomware Programs/Scripts:** Actual ransomware programs or scripts can generate realistic attack data.
- **Emulator:** A ransomware emulator (e.g., WannaLaugh) can generate synthetic ransomware activity for controlled experiments.
- **Other Tools:** Additional tools (e.g., memory monitoring, specialized malware analysis frameworks) can be integrated to gather a comprehensive dataset.

### Controlled Environment

- **Controlled Setup:** All file operations occur in a single directory (e.g., `./data`) to reduce noise and focus the experiment.
- **Synthetic Ransomware Activity:** Simulate ransomware behavior using an emulator or scripts that mimic ransomware actions (e.g., encrypting files, establishing C2 communication).
- **Full Data Collection:** Data is collected from all relevant sources (file system operations, process calls, network traffic) to create a rich, multi-faceted dataset for ransomware detection.

## Data Collection and Preprocessing

- **Comprehensive Data Collection:**
    - **Auditd Logs:** Captures detailed file I/O operations and process activities.
    - **Strace Logs:** Captures system call traces, especially file and network-related calls.
    - **Suricata Alerts:** Captures network traffic anomalies, such as C2 communication, unusual DNS queries, and large outbound traffic spikes.
- **Synthetic Ransomware Activity:** Simulate ransomware programs to generate realistic attack data (e.g., salami-slice method). Augment the logs with metadata or annotations about known ransomware activity.
- **Feature Extraction:** We aim to minimize handcrafted feature engineering by leveraging deep learning to learn features from raw data. However, some preprocessing (e.g., structuring system calls) may be necessary.

## Ransomware Detection Approach

- **Sequence Prediction Model:**
    - **Model Type:** A sequence prediction model (e.g., Transformer) that learns temporal and sequential patterns from a multi-source dataset (file system operations, process activities, network traffic).
    - **Training Data:** Labeled sequences from the combined dataset, augmented with metadata indicating whether the sequences represent normal or ransomware activity.
    - **Evaluation:** Evaluate the model’s ability to detect ransomware based on the full dataset from multiple sources, assessing both false positives and false negatives.

- **Ideal Environment for Model Training:** In this research setup, there are no constraints on computational power or resource usage, allowing for deep exploration of model architectures, training techniques, and hyperparameter tuning.

## Research Hypothesis

**Hypothesis:** By leveraging a comprehensive dataset that includes file system operations, process activities, and network traffic, a sequence prediction model can effectively detect a wide range of ransomware activities. The integration of multi-source data enables the model to learn subtle patterns of malicious behavior, distinguish them from normal operations, and provide predictions and explanations for potential ransomware activity.

## Experimental Plan

1. **Data Collection:** Collect a comprehensive dataset from `auditd`, `Suricata`, `Strace`, and other tools. Include both normal operations and simulated ransomware attacks, with annotations to indicate known ransomware activity.
2. **Model Training:** Train a sequence prediction model on the combined dataset, focusing on learning patterns across multiple domains (file system, process calls, network).
3. **Evaluation:** Test the model on new data to evaluate its detection accuracy on out-of-distribution data.
4. **Refinement:** Continuously refine the model architecture, feature extraction, and dataset based on experimental results, exploring different techniques to improve detection accuracy.

## Production Considerations (Future Work)

- **Real-World Constraints:** Future work will involve adapting the system for real-time production environments, where computational and resource limitations may require optimizations.
- **Simplified Models:** Explore ways to deploy more efficient models in production, possibly by leveraging lightweight versions of the sequence prediction models trained in the research environment, such as using distilled models.

## Conclusion

Insights gained from this research will inform the development of ransomware detection systems that can proactively identify malicious activity based on the way data is updated. By combining data from multiple sources and training a sequence prediction model, we aim to achieve high detection accuracy with minimal false negatives. The research will contribute to the ongoing efforts to combat ransomware threats and enhance cybersecurity practices.

## References

WannaLaugh: A Configurable Ransomware Emulator. Link: https://arxiv.org/abs/2403.07540
