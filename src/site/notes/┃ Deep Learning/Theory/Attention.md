---
{"dg-publish":true,"permalink":"/deep-learning/theory/attention/"}
---


---
# What is Attention?

Attention is a mechanism used in the [[┃ Deep Learning/Theory/Transformers\|Transformers]]' Architecture which makes the model be capable to focus on various parts of the sentence at the same time to determine the importance of each word in the input sentence.

$$\Huge\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

---
## Q, K, V?

Now that you see the equation, you are going to ask yourself what these letters mean. 

 - Q represents the query, per example the question that you want the model to ask.
 - K represents the key, which is the result of the hidden layers from the encoder, the model uses it to find the important data and its correct value associated.
 - V represents the value, per example, in a translation task, the value represents the words in the language that you want to translate to.

---
## [[┃ Deep Learning/Theory/Softmax Normalization\|Softmax Normalization]]

In simple terms, the softmax function is used to normalize the values of the outputs, this means the sum of all values will equal 1, therefore, converting them into probabilities to choose the most accurate solution based on the model calculations.

**For the softmax function:** [[┃ Deep Learning/Theory/Softmax Normalization\|Softmax Normalization]]


$$\Huge\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$$







