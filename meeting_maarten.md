# Notes
- gpt-2 has a window size
     - maybe segment by some window size, paragraph, etc. for PAPI calls
     
- get a PerspectiveAPI clone and try running labels through it...
    - can we just get a pretrained model, or do we need to train our own
    
- would be cool to just have a toxicity distribution 

- there are known issues with false positives...
    - cite Maarten's paper!
    
- different styles of language
    - easier to find gold labels for websites
    - can you actually use websites as labels, since a single source could have different biases (eg. right or left leaning NYT opinion articles)
    - 


# Next Steps
1. Train a classifier in the same way as Perspective API
    - https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
    - https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
    - VAMPIRE or... maybe BERT
2. Using clasisifed OpenWebText...
    - train two models, one with Toxic Language and one without
    - trained or pretrained (depending on amount of compute we have)
    - see the incidence of toxic language in the outputs (holding context constant between predictions)
