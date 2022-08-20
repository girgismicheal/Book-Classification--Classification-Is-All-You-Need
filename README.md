# Book-Classification--Classification-Is-All-You-Need

# Gutenberg Project Overview
**Project Gutenberg is a library of over 60,000 free eBooks**

![Gutenberg](https://drive.google.com/uc?export=view&id=1bOd8Hiv-sU8Skj1gYR-2cxLUEBIretyZ)

In this project we selected some books from Gutenburg library from different categories and then select random paragraphs from then and labeled this paragraphs by the book name for ground truth.
After creating the dataset we used many transformation algorithms to embed the text to numbers for modeling process like (Fast-text,BERT ,TF_IDF,BOW,Skip gram ,Glove ,LDA ,Word2Vec ,Doc2Vec)
<br> After this we tried many clustering algorithms like(K-means,Expected-maximization(EM),Hierarchical,) and chose the champion one which achieved the kappa and silhouette score.


**Recommended to use GPU to run much faster.
But it works well with the CPU also.**
- GPU takes around 40 min, while CPU may take hours.

# Project Methodology

![Gutenberg](https://drive.google.com/uc?export=view&id=1-YX8_vqTOSjudFe5AiovLCDzA6SEOhLm)
