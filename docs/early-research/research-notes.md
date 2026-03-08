
Since we have access to nearly every form of ransomware threat, we can generate a *significant* amount
of data to train detectors on.

We can approach this in a few different ways.

## Sequence Prediction

- One way is as a sequence prediction problem. Train a model to predict the patterns in the data of a
ransomware attack, along with metadata about the attack.
and once it has a good predictive model, this model can be used to predict the attack based on the
observable data, potentially predict other kinds of queries too, like what kind of attack it is (metadata).

- I think this may be why Fujinoki was interested in my mention of process supervision data. Not only
do we have a *lot* of real data here, but we can generate as much new data as we want. We can potentially
even train it to predict how to generate new python programs to generat enew variants of ransomware attacks.
These new automatically generated variants may be potentially useful as new sources of synthetic data.

- for the sequence prediction, both patterns in the ransomware programs and patterns in the dynammic
run-time behavior of the ransomware can be predicted/detected.


## Classification

There is also a lot of opportunity to simply generate classifiers based on deep NNs. This isn't about
sequence prediction, but input -> predict class. The downside is that it relies on more labeled data,
which may not be a problem if enough data is present or can be synthetically generated.

There's this idea in deep learning that anything a human can do fairly quickly can be done by a
deep NN. Can an expert in detecting ransomware do this pretty quickly if given the data? Probably in many
cases, but I suspect that not always. Let me explain: any human can look at a very complicated image and
immediately detect whether a cat is in the picture. Takes way less than a second. DNNs can do this too.
If we put the data representing a ransomware program in front of an expert, say a visual representation
of the prorgam code or run-time behavior, could the expert immediately detect the presence of
ransomware like we can a cat? If so, maybe ransomware detectors can simply be built as a classifier:

  data/evidence -> probability distribution over type of ransomware (or None)
  data/evidence -> {yes, no}

i suspect this would be pretty effective. i actually made a program in 2013 based on a simple NN
that could detect a large set of viruses reliably. it would output not a class, but a probability
distribution over the classes. related to fujinoki's desire to see a negligible false negative rate
and a small false positive rate, this NN program i made also came with a cost matrix that factored
in he severity of the type of virus detected and the probability, e.g., a low probability but severe
outcome -> report it, but a low probability and less severe -> don't report. the reason i did this
is the same reason fujinoki is probably worried about it: if the false positive rate is too high,
it'll be turned off. there is a trade-off between false positives and false negatives: you can
make the false positive rate approach 0 if you allow false negative rate to approach 1.


## Benefits of Sequence Prediction

An autoregressive model like chatgpt (they append their output to the front of their next input (context)
to generat the next output token, rince and repeat) generally can be understood as trying to learn
an approximation of the underlying data generating process (DGP) that led to the data you trained it on.
Later, of course, we can fine-tune it for special tasks, like detecting ransomware. The nice thing
about knowing the DGP is that once you know the DGP, you essentially know in a sense everything there
is to know about the data. Classification itself is just an easy special case:

Pr(class|data) = Pr(class, data) / Pr(data)

It also benefits from being able to learn from *raw*, unlabeled data. The only "labels" are given
implicitly by the order of the tokens, e.g., in a causal LM we try to predict the next token
based on the previous (causal assumption or at least correlation based on the past) tokens, this is
called self-supervised learning, but it's really just learning from the raw data by predicting
the next token. This same principle applies to any modality, whether the tokens come from, say,
a camera sensor or language.

#### What I'm Leaning Towards

Data:
   1. File system behavior
   2. Network traffic behavior
   3. API call behavior (this one is apparently already pretty popular, according to the survey, and has
      been used to train sequence prediction models like RNNs and LSTM. if we scale this up and generate
      (or collect) for more data of various kinds, it may work quite well. we could use it on a transformer
      architecture though, since that's where most of the activity is right now.

a lot of this data is very easy to tokenize, and in many ways suitable for even existing language
models (LMs). maybe we could even fine-tune an open source model (or even GPT-4o? pretty cheap to fine-tune those models. i have some experience generating data for fine-tuning models, see my github repo about fine-tuning an LM to be better at the task of mapping natural language queries to ElasticSearch DSL JSON queries.

i suspect it would be state of the art if it hasn't already been done. that's a recurring pattern.

**Goal**: negligible false negatives, small positive rate. since we migth use a language model, while
we can have it output a class label (and even fine-tune a version of the model to do this task), we can
also have it be able to answer questions about it if we include such linguistic information in the training data, or correlate such linguistic data with the training data in some way that the LM can latch onto.

of course, we want a quick classifer to immediately respond to threats, too, which is why a fine-tuned model (or a DNN classifier trained from the ground up) is still a necessary idea.
