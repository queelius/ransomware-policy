# Reading Notes on `rans_survey.pdf`

## Section 1: Introduction

- Ransomware-as-a-service (RaaS) model would seem to represent a new level of sophistication and acccessibility for ransomware attackers.

## Section 3: Ransomware and Evolution of Ransomware

- Two kinds of ransomware: **Locker** and **Crypto**. Locker is relatively harmless, while Crypto is the real threat. We focus on Crypto ransomware detection in our research.

- Based on Fig 2 in Section 3, new ransomware families are emerging more frequently, indicating a growing threat landscape. This is likely due to the increasing profitability of ransomware attacks

- Our research focus on AI-driven ransomware detection will likely focus on observable patterns in the infection, C&C,and destruction phases. The early something can be detected in this sequence, the better the chances of preventing data loss/corrution.

- Cryptocurrencies led to the surge in ransomeware, as it makes it possible for them to make the ransom without
too much fear of getting tracked down. However, as the survey points out, blockchains are only pseudo-anonymous.
More importantly, maybe some AI detection can be used on blockchain transactions to detect or predict
ransomware activity. This may allow for a more systemic defense that cuts across businesses. A similar argument can be made for payment/extortion frontends hosted on Tor.

## Section 4

- In Domain generation algorithms (DGA) that automatically generate many domains periodically, they may generate gibberish names. Probably easy to detect, thus they probably instead use coherent names drawn from word lists and such, or even use LMs.
    - Wikipedia: Recent attempts at detecting DGA domain names with deep learning techniques have been extremely successful, with F1 scores of over 99%.
      https://en.wikipedia.org/wiki/Domain_generation_algorithm

- More on motivation: even if infinite backups (see reliable backup assumption doc) happens at a very fine
granularity so that no data can ever be lost, the attacker can still extort (threaten to release private data
for instance). I only mention these ideas because I want to motivate the need for the research without relying
on potentially unrealistic assumptions.

## Section 5

### Section 5.1

- Static analysis: they mention obfuscation and such to mitigate static analysis techniques. I believe
Fujinoki and I have both did a lot of research on obfuscation techniques. I have a large body of older
research where I explore ideas in oblivious computing.

- Dynamic analysis. The author of the paper claims that concealment techniques cannot be effective against dynamic analysis since those approaches cannot conceal the behavior of ransomware. I might push back on this claim. I have did a lot of research on enabling oblivious computing, and making program inputs and outputs look essentially like noise in some cases. However, utlimately, a useful oblivious program would need make some system calls to access or write files, at which point a dynamic analysis would indeed be effective and the concealment not possible.

- But, even so, a kind of stenography for hidden computation and program obfuscation is possible to some
extent such that a lot of the activity can be effectively concealed even to a dynamic analysis. These
oblvious computing techniques could potentially be used by, say, a ransomware attacker to try to evade
detection or prevent researchers from learning as much about the attack.

### Section 5.2: Ransomware Detection Research

- Most of these eight categories seem like non-starters.
- Blacklists won't be effective given all of the new techniques and sophistication of attackers (e.g., DGA).
- Rule-based systems: probably going to be too hard to codify a set of rules that will provide robust detection. I operate from the intuition that the patterns and evidence left behind by these attackers will be too subtle, too polymorphic, to detect with easy solutions that we can work out with reason. The same reason we can't make a program to detect a cat.

- information-theroetic: this does actually seem reasonable, in theory, but it's not clear it'll be useful in practice. as they point out, encryption and compression would both trigger these methods, and these are very widely used. so, a lot of false positives.

- Looking at the table 2 that reports TPR and FPR, i noticed the TPR is quite high (very low FNR) on
identifying threats. i agree with Fujinoki that i'm skeptical, but previously i mentioned that
i was able to geneate a NN model that very accurately classified viruses. so, maybe it was possible,
and maybe it still is. however, there's every reason to suspect that the sophistiction and
detectability of these ransomware programs will increase and older detection methods will not be
sufficient.

- 100% TPR is overly optimistic though, agree with both Fujinoki and the author of the survey. I would
question their testing methodology. i will look into some of their papers and testing methods
to see if i can make any sense of it.

- for instance, maybe they tested on the same data they trained on. in that case, the reported results
are relatively useless. ML models can very easily overfit to the data.

- i just read in the survey that the majority of the studies *did not* report any data sources for their
evaluation datasets. this is a red flag.

