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
