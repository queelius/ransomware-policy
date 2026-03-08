## Report: Detecting Ransomware Threats Using Predictive Models

With access to extensive datasets (either collected or generated algorithmically)
of ransomware attacks, there is an opportunity to trainable models that can
roboustly predict ransomware threats. Predictive models, such as large language
models (LLMs), offer a promising approach to identifying and explaining
ransomware behavior by analyzing various features of the data.

### Skepticism of High Evaluation Benchmarks in Ransomware Detection Studies

In reviewing the ransomware survey, several concerns align with Fujinoki’s skepticism about the high evaluation benchmarks reported in some studies:

1. **Unclear Data Sources**:
   - Many studies didn’t report where their evaluation data came from. This lack of transparency raises doubts about the results, as it’s unclear whether the data was representative or if the models were simply overfitting.

2. **Possible Overfitting**:
   - If models were tested on the same data they were trained on, the reported high performance is not meaningful. Overfitting to training data can make the results look better than they are in real-world situations.

3. **Overly Optimistic TPR and FPR**:
   - Some studies report nearly perfect True Positive Rates (TPR), which seems unrealistic. Given the complexity of modern ransomware, it’s doubtful that detection methods are as effective as claimed without also having a high False Positive Rate (FPR).

Overall, I agree with Fujinoki’s doubts about these benchmarks.

### Predictive Models for Ransomware Detection

Predictive models, like LLMs, involve training a model to forecast certain features of the data based on other observable characteristics. In causal models, this typically means predicting future observations based on past data. For ransomware detection, predictive models could be trained to anticipate the presence of ransomware by analyzing indicators such as file system behavior, network traffic, or API call patterns. Once a robust predictive model is in place, it can be employed to predict, identify, and even explain ransomware attacks in real-time.

#### Advantages of Predictive Models

- These models can predict not only the presence of ransomware but also specific attributes of the attack, such as its type and severity.

- Predictive models essentially learn an approximation of the underlying data generating process (DGP), making them quite versatile, e.g., classification tasks are just a special case:
$$
\Pr(\text{class}|\text{data}) \propto \Pr(\text{class}, \text{data})
$$

- These models can generalize across different types of data, from static code analysis to dynamic run-time behavior, offering broader coverage in ransomware detection scenarios.

- **Minimal Labeling Requirements**: The ability to learn from raw, unlabeled data significantly reduces the need for extensive labeling, enabling the model to train on large datasets with minimal preprocessing.

### Synthetic or Algorithmic Data

- Apply known algorithms (including the ransomware programs themselves and plausible variations) to generate data simulating a wide range of ransomware behaviors.

- This may expand the training datasets and improve their ability to generalize to new variants.

- Synthetic data is particularly valuable when real data is limited or when exploring new types of ransomware that have not yet been observed in the wild.

- Data such as file system behavior, network traffic, and API calls can be easily tokenized. This allows the possibility of fine-tuning existing language models on ransomware data, leveraging the strengths of pre-trained LMs.

### Process Supervision Data

- **Let's Think Step-by-Step**: Process supervision involves augmenting data with reasoning steps or rationales, guiding the model in predicting and explaining the underlying processes. For ransomware detection, this could mean breaking down an attack into interpretable steps, enabling the model to both detect the threat and explain its progression.

- Automatically generating process supervision data presents new opportunities to enhance the model's interpretative capabilities, making it more robust, explainable, and capable of providing rationales for its predictions.

### Research Goal

The primary goal of this research is to achieve a negligible false negative rate while maintaining a small false positive rate, validated through robust evaluation datasets. By leveraging language models, possibly through fine-tuning existing models, this research aims to develop a system capable of detecting ransomware with high accuracy while also offering explanations and rationales for its predictions.
