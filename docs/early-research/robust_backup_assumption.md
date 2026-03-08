# Research Assumption: Robust Backup System

For the purposes of our research, we assume the existence of an ideal backup system with the following characteristics [^1]:

1. Infinite Rewindability: The system can be rewound to any point in time.
2. Read-Only History: All historical data is immutable, preventing tampering or corruption.
3. High Reliability: The backup system itself is not susceptible to failure or compromise.
4. Fine Granularity: Backups are made frequently, capturing incremental changes.

## Importance of Detection Despite Ideal Backups:

Even with this robust backup system, ransomware detection remains crucial due to:

1. Data Integrity: Malware may corrupt data over time, making it difficult to identify a clean restore point.
2. Infection Point Identification: Determining the exact moment of infection is challenging, complicating the choice of which backup to restore.
3. Incremental Corruption: Subtle, gradual changes may result in all backups containing some level of corruption.
4. Legitimate vs. Malicious Changes: Distinguishing between normal data evolution and malware-induced changes is complex.
5. Recovery Complexity: Restoring from an old backup may lead to significant data loss or inconsistencies.

## Core Research Focus:
By assuming an ideal backup system, we emphasize that the challenge lies not in the backup mechanism itself, but in the timely and accurate detection of ransomware activity to minimize data corruption and facilitate effective recovery.

For details on our specific research focus, please refer to the `research_focus.md` file. This separate document outlines our rationale for focusing on cryptographic ransomware and the key objectives of our AI-based detection methods.

[^1]: This assumption is made to focus on the detection aspect of ransomware attacks, rather than the backup mechanism itself. This is a common practice, e.g., in cryptography, assumptions like the random oracle model are made to simplify the analysis.

