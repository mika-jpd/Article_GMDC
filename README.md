# WebsCrapping-and-GMDC

Work in progress!

The goal of this project is to create a Webscraper to scrape news articles from the NPR website and then classify the articles into different clusters. 
This program uses KMeans algorithm to produce an initial clustering. This will serve as the initialization clusters for the Gaussian Mixture document classification model.

# Current issues:
* numerical stability during caculations of the diagonal covariance matrix
* converting the text into a panda Data Frame tends to produce null values, this is probably a consequence of insufficient text cleaning

# Paper
The GMDC clustering model is based off A Murua et al. which compared different document classification methodologies. The full article can be found [here](https://sites.stat.washington.edu/people/wxs/Learning-papers/MuruaStuetzleTantrumSieberts-A4.pdf).
NPR articles are scraper from there politics section [here](https://www.npr.org/sections/politics/)
