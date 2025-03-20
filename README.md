This repo shows how to use machine learning to win about 77% of Hangman games on a subset of 5K words from a collection of [479K English words](https://github.com/dwyl/english-words/tree/master). 

Many of the words in the corpus are pretty bizarre. While playing with friends on randomly chosen words, we won around 30% of the time. I'd suspect on a larger subset, we would have done
better but probably not above 50 - 60%. 

The strategy used is:
(1) Use a frequency-based manual strategy on words from the training set. This approach encodes linguistic patterns and approximates the value Pr(next_letter | game_state = (word_state, incorrect_guesses, lives_left))
(2) Encode those (game_state, manual_prediction) into torch tensors
(3) Train a shallow but deep transformer model on those tensors
(4) Play with the trained model, ideally masking previous guesses to improve performance

Training took around 12 hours on a v4-8 TPU generously provided by Google's TPU Research Cloud Program.

I'm not including the weights or the exact word list used, although it is a subset of the words linked above, for two reasons: first, this challenge was given to me by a company, and to 
maintain some semblance of fairness, I won't post the weights or the exact word list; second, the weights file is very large.

