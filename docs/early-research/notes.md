<refer to my SLUUG talk, too>

STaR: Self-Taught Reasoner - Bootstrapping Reasoning With Reasoning
https://proceedings.neurips.cc/paper_files/paper/2022/file/639a9a172c044fbb64175b5fad42e9a5-Paper-Conference.pdf

I talked with Dr. Z over zoom about these ideas several months ago. He had just been hired at x.ai out of stanford,
so he was under NDA and prevented from talking about his more recent work, but we had a very interesting
conversation about how to bootstrap these models to generate more "system 2"-like reasoning.

one of the proposals i put forward that he thought sounded promising and that he could talk about without NDA
restrictions was based on some of the ideas you and i have been discussing. essentially, how to generate
good training, synthetic or algorithmic, training data with graduated difficulties and step-by-step rationales.
my idea in particular was this:

many problems are extremely hard in one direction, but easy in the "opposite" direction. i think the most simple
example of this is solving non-linear differential equations, versus staring with a function and taking
partial derivatives to go from solution -> problem. the opposite, problem -> solution, is much harder.
the simplest case is:

y'(t) given, like y'(t) = 2t
y(t_0) given, y(0) = 0


then, we are tasked with finding y from the given information.

if we START with y(t) = t^2 + c, then we can just take the derivative, y'(t) = 2t.
for more complicated expressions, there are a lot of rules that need to be applied to symbolically solve the
problem, but it's still straightforward if we're just starting with a "solution" and applying partial derivatives,
e.g.,

y(t) = exp(log(t) + t^2 + 1) - cos(log(t)) + 1
y(1) = exp(log(1) + 1 + 1) - cos(log(1)) + 1 = exp(2) - cos(0) + 1
     = exp(2)

y'(t) is just computing using the following sequence of steps:

step 1: ...
...
step n: y'(t) = exp(log(t) + t^2 + 1) ( 1/t + 2*t ) + sin(log(t)) (1/t)

now we reverse the problem, going from step n to step 1, and ask
what is y(t) given y(1) = exp(2)?

we have the step by step solutions going from step 1 to step n. we reverse it.
we can annotate each of these steps using a symbolic solver. we can create problems that are as difficult
as we like. if we were to go from problem to solution, it may require a significant amount of backtracking
and clever pattern recognition to be able to make progress to the solution, but since we're *starting*
with the solution and working backwards towards a starting problem, no clever insights are needed. we only
need to apply well-known rules in symbolic differentation one step at a time. each time we apply the rules,
we can provide a rationale (automatically). then, when we *reverse* these steps, we reframe it as not a
differentation step, but an integration step. we can even use language models to generate natural language
descriptions of these steps in reverse.

we can create *as many problems* as we want, that can be as difficult as possible (while still having
a closed solution, or a symbolic solution). then we have the language model predict these solution steps
over the annotated, algorithmically training data.

we can do this for partial derivatives, we can do it for solving n-th order problems, and eventually,
we can even incorporate numerical solution steps and maybe "visual" analysis, teaching it how to
generate graphs of the problem, identify singularities, and so on.

this will likely work QUITE well for this kind of problem. the neat thing is, a lot of problems
take this form of being hard in one direction, but easy in the other. NP problems, for instance,
in general may have this characteristic, where it is easy to verify a solution but hard to generate
a solution. the trick is to find a way to reverse the process. i have also started to devise a way
to do this for theorem proving in general for arbitrary rules. we don't start with a theorem to prove,
but we instead start with proof steps by doing random walks to generate a graph where nodes are
expressions and edges represent applications of rules. now we take any two expressions connected by this
graph, and enumerate the proof steps to get from the starting expression to the ending expression.

if we give it a lot of different sets of rules and for each set of rules, give it a number of these kinds
of problems to predict. some of these rule sets will be well-known, like calculus or algebra, but
we can also make up rules. maybe we find that it can not only be trained to generate solutions within
the well-known rule sets, and thus have immediate value in solving real problems, but by giving it
arbitrary rule sets, it may have a kind of meta-learning on how to reason over reasoning and planning
over any set of rules.

now, let's think about how we can do this in particular for ransomware.

## Language model approach

first, we can start with real data. assuming we're using a sequence prediction model, like an LLM,
we tokenize the data and represent the data as a sequence of tokens. normally, the sequence
will represent harmless data. sometimes, however, it may represent ransomware activity. during the
pretraining stage, where we just learn from raw data, it may not be important to *label* this
data as indicative of ransomware. we can simply have it predict the data, including ransomware
activity data.

maybe it is the case that with a instruction-turned LLM, we can actually just *prompt* it with
many examples of ransomware, and ask it:

"Given the previous examples of ransomware activity, does the following data look like ransomware?
Let's think step-by-step. Please explain your reasoning."

If this works well, and sometimes it does, pretraining a new model or fine-tuning an existing model
may even be needed. that's an empirical question. that's what i can do initially, just to see if
it works well, and if nothing else, give me a baseline. this will also be my opportunity to
come up with some evaluation benchmarks. i imagine there are already a lot of such benchmarks for
ransomware identification, but we have the added benefit of rationales that the LM can produce
to justify why it is or is not ransomware activity. this may call for a new set of benchmarks where
we also care about the rationales, not just the accuracy of the "final" output.

so, next step may be to fine-tune or train a LM to do the same task. the nice thing about fine-tuning,
for instance, is that we can rely upon their very broad competency in a wide range of linguistic
tasks (predicting linguisic tokens produced by humans and the tools they use). then, on top of that
core LLM, we can fine-tune it on labeled tasks, i.e., on ransomware activity, labeled in some way
using natural language, e.g., we give it some ransomware activity, and we take any or all of the following
approaches:

1) a class label, like "<tokenized ransomware activity>. prediction: ransomware."
it can also be more fine-grained, like the type of attack.

2) seek out labeled ransomware data. fine-tune or pretrain a LM on it. ideally, the data will include
rationales and other reasoning steps that explain why it is believed to be ransomware, but even if it
doesn't, it can still be valuable data. the language model itself can generate rationales based on its
more "common sense" knowledge, as in the training data there will still be a lot of data that correlates
ransomware activity with ransomware identification.

for instance, a fine-tuning data set with rationales might look like "<tokenized ransomware activity>.
in order to determine if this is ransomware,
let's look more carefully at the data. first, we see that there was a file operation where some
seemingly important documents, for example "my-quarterly-report.txt", was modified in such a way that its
entropy drastically increased. if it was just compressed, the entropy may be approximately the same,
but with a smaller file size, but in this case, the file size is larger. it looks like it may have been
encrypted. many other important documents were also seemingly encrypted. at the same time, some suspicious
traffic was identified. for instance, ..."

3) whatever data like this already exists, we can collect it and pretrain or fine-tune a pretrained model
on the data. the nice thing about fine-tuning vs prompting is that the fine-tuning data is not limmited
to some fixed-size prompt, as is the case when prompting. we can just fine-tune it to be generally more
capable of predicting this SUPERVISED task of identifying and providing rationales for ransomware activity.
we can still also prompt it. this is likely how we would compare:

a) prompting only, few-shot examples. maybe using other inference algorithms, like sampling N separate
outputs and majority voting on the class label. (we can have more recent models strictly adhere to
an output format, like a JSON, which can include not only field names, but a field values. these are
known as grammar constraints on the output of LMs, and they can be thought of as generating outputs
over a conditional distribution that requires the output tokens meet certain conditions. beam-search and
other techniques may be used. we can also explore more complicated methods, like tree-of-thoughts,
or even more agentic-like frameworks where we have multiple independent sessions of an LLM engage in
"talk" with themselves to try to figure it out.

b) fine-tuning only. no prompt. just ask it, after fine-tuning it.

c) prompting + fine-tuning

3) we may now want to see if we can improve some metric (eval) over (1), (2), and (3), by generating
algorithmic/synthetic data. (we could also take a very large language model, and use it as a mentor
to train a much smaller model on supervised training data of interest by having the much smaller model
try to reproduce the LOGITS of the much larger model. this often works quite well, getting most
of the performance of the large model on the given task in a much smaller model. the smaller model
may be needed if these models are running non-stop in the background to reason about the tokenized
data as being ransomware.

4) finally, we may want to generate algorithmic/synthetic data. this is where a *LOT* of potential
exists. we can take existing ransomware programs and run them to generated synthetic data.
since we're generating the data, we can also label it as ransomware/not ransomware. we can run
a lot of existing ransomware programs, and also even train the LM on all kinds of existing solutions
for ransomware programs, and then have it predict plausiblle vartions of these programs, so that
we not only generate data about existing ransomware attacks, but give it insight about plausible
future kinds of attacks (out-of-distribution generalization) or attacks not identified yet (but
may already be rampant).

5) since we're generating the labels in (4), i want to think very carefully about how to generate
rationales for why it identifies it as ransomware. what is the evidence? what is the justification?
why might it be wrong? (false positives or false negatives)

since we know the "ground truth" (what program are we running), we can 
	


6) a big reason prompting may not work well -- but this is pure speculation:

   - a ransomware attack may take place very slowly (you mentioned one of these ideas in the survey)
over time, and the evidence mmay be quite sparse and spread out over time. when we use prompting,
we fill the context window with the sequence of tokens that represent some subset of activities
on the computer or network. but, most of this data is harmless, and only a very small proportion of
the data may represent the actual attack. that is to say, there is a low signal-to-noise ratio,
and it's not obvious that a LLM will be able to just put all of this activity in the context
and produce reliable outputs.

   - thinking about it, i see two solutions for the prompting only appraoch:

      - solution 1: have the LM produce intermediate outputs over time. every 5 minutes, for instance,
        have it analyze the data in that time window and essentially summarize the data with respect
        to indicators of ransomware. instead of storing all of the tokens, store these tokens (we can
        keep both, of course, but the idea is that we have it extract potential signals from
        time segmetns of data, and have it reason over this much smaller and more salient signal.

     - use a very large context window and just hope it can do it. this is the first thing
       we can try. context lengths are getting larger, and may avoid possible problems with the
       above approach that may miss correlations that are spread out over much larger spans than
       5 minutes. the best of both worlds is to do both, where they can mutually inform each other.



## DNN approach

We can forgo the LM approach and go for a simpler DNN approach, data -> label


a nice gthing about the LM approach is we can also have the model *predict* actions for the defense
to take to mitigate the harm from the potential ransomware attack.

## RL approach using self-play

This is an ambivious approach, and also hardest. think alphastar. aligning the ransomware adversary
with real-world-like ransomware activities may be hard. it kind of requires us to 
