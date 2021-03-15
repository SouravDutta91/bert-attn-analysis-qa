# Whatcha lookin'at? DeepLIFTing BERT's Attention in Question Answering
Analysis of Attention in BERT for QA Explainability. There has been great success recently in tackling challenging NLP tasks by neural networks which have been pre-trained and fine-tuned on large amounts of task data. In this paper, we investigate one such model, BERT for question-answering, with the aim to analyze why it is able to achieve significantly better results than other models. We run DeepLIFT on the model predictions and test the outcomes to monitor shift in the attention values for input. We also cluster the results to analyze any possible patterns similar to human reasoning depending on the kind of input paragraph and question the model is trying to answer. We are using the SQuAD 2.0 dataset for this task.

## Dependencies

`pip install pytorch-pretrained-bert`

## Model

For our experiment, we had used an available BERT model pre-trained on the SQuaD 2.0 dataset. You can use any such pre-trained BERT model available online or train your own model if you have enough resources.

## Run the script

Run the file `code/script.py`.

## Citation

If you are referring to this work in your research, please cite as:

```
@article{arkhangelskaia2019whatcha,
  title={Whatcha lookin'at? DeepLIFTing BERT's Attention in Question Answering},
  author={Arkhangelskaia, Ekaterina and Dutta, Sourav},
  journal={arXiv preprint arXiv:1910.06431},
  year={2019}
}
```
