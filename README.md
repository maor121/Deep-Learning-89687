# Deep-Learning-89687
Deep learning for Texts and Sequences course at Uni, repository for assignments

Language: Python <br/>
Framework: PyTorch <br/>

Assignemnt 1
------------

Assignment 2
------------
**Window based tagging**<br/>
Solve 2 common problems in NLP: POS (part of speech) tagging & NER (named entity recognition).<br/>
Do this using several approaches, model is always a single layer MLP (multi layered perceptron), with input of constant size: a window<br/>
1. window of words, every word is an id.<br/>
2. window of words, every word vector is the word embedding (pretrained)<br/>
3. window of words, every word vector is the sum of it's: (a) word embedding. (b) suffix vector (c) prefix vector

Assignment 3
------------
**RNNs and BiLSTM**<br/>
Solve POS and NER using BiLSTM (bidirectional LSTM). Also, explore RNN limitations, what kind of sequences are easy for LSTM to learn, what is difficult?<br/>
Challenges included: batching for sequences (non-uniform length).

Assignment 4
------------
**Implement an SLNI  paper - An exercise in Attention**<br/>
NLI (Natural Language Inference) is a problem in NLP in which given two sentences, the computer needs to know whether they: (a) contradict each other (b) neutral (c) they agree.<br/>
The SNLI challenge is an NLI challenge managed by Stanford Univesity, and recent papers managed to get 89% accuracy (2018).<br/>
The paper I chose (https://arxiv.org/abs/1606.01933), solved this problem using a relatively intuitive approach:
**High level description:**
1. First, each sentence is converted to it's word embeddings (Glove pretrained).
2. Every pair of sentences are softly aligned to one another. Meaning every word from sentence 1 is softly aligned to all the words in sentence 2. Then you concat: every word from sentence 1 with subphrase from sentence 2.
3. the concatination of word and subphrase is passed to a MLP which determines weather they "agree", "contradict", or "neutral" with one another.
4. The result is summed, thus you "count" how many parts in the sentences "agree", "contradict" or are "neutral" with one another. And the result is this passed through a softmax to get the discrete class.<br/>
**The Pros of this approach:**
1. Intuitive - much simpler and cleaner then other approches.
2. Quick to run - this is an Attention without LSTM. LSTMs are knowns for their long learning times, and with how much data they need.
3. High accuracy - 86% (paper), 81% (My own)
