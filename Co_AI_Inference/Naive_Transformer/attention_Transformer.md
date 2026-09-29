For a given token, the model assigns attention weights from that token to every token (including itself) in the sequence. It then uses these weights to create a new representation for that given token as a weighted sum of the existing representations of all tokens. This enables the model to dynamically focus on the most relevant information (the tokens with the highest weights) at each step and capture complex, long-range dependencies in the data.


To recap, when realized within a transformer, the attention mechanism is powerful for *three key reasons*: 



*Handling of long-range dependencies*: earlier sequential modeling techniques frequently found it challenging to establish connections between widely separated words. The attention mechanism overcomes this by creating a direct link. This enables the model to identify and utilize information from a word near the start of a text as essential for interpreting a word much later, preventing the informational signal from weakening across numerous steps.



*Parallelization*: in contrast to previous architectures that analyzed tokens in a step-by-step, linear fashion, the attention mechanism allows for a large portion of its computations to occur simultaneously. Leveraging specialized parallel hardware, such as GPUs, for these simultaneous operations makes it practical to develop and train significantly bigger and more capable models.
 

*Deep contextualization*: the mechanism of attention grants each input token the ability to directly access and integrate data from all other tokens within that input. Consequently, the internal representation of a word is not stand-alone but is shaped by the surrounding context of the sequence.