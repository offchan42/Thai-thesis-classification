import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def pretty_trim(text):
    words = text.split(u'|')
    stripped_words_generator = (word.strip() for word in words)
#     stemmed_words_generator = (stemmer.stem(word) for word in stripped_words_generator)
    trimmed_words = (word for word in stripped_words_generator if 1 < len(word)) # retains words that are not empty
    alpha_words = (word for word in trimmed_words if not word.isnumeric() or len(word) <= 4) # allow only <= 4-digit number
    return u' '.join(alpha_words)

def simple_split(string):
    return string.split()

"""
**k** can be a float to represent minimum accumulate sum of the probability, or an int specifying constant number of predictions
If **k** is an integer, it will be the constant number of predictions to make for each sample

If **k** is a fraction, it will be the minimum confidence score.
The model would automatically choose different number of predictions for each sample.

For example, if a model is very confident that 'X' should be assigned to class 'Y' or 'Z' with the probability of 50% and 30% respectively then it would need only 2 predictions to do the job if you specify **k** to be <=  _0.80_.
"""
def score_top_preds(clf, X, Y, k=1, plot=True):
    prob = clf.predict_proba(X)
    top_probs = np.sort(prob)[:,::-1]
    top_predictions = prob.argsort()[:,::-1]
    top_probs_cumsum = top_probs.cumsum(axis=1)
#     print 'Top Probabilities'
#     print top_probs
#     print 'Top Predictions'
#     print top_predictions
#     print 'Top Probabilities Cumulative Sum'
#     print top_probs_cumsum
    if isinstance(k, float):
        needed_preds = (top_probs_cumsum >= k).argmax(axis=1) + 1
        correct = np.empty_like(Y)
        confidence = np.zeros_like(Y, np.float32)
        for i in xrange(X.shape[0]):
            predict = np.array(top_predictions[i,:needed_preds[i]])
            correct[i] = (predict == Y[i]).any()
            confidence[i] = top_probs_cumsum[i, needed_preds[i]]
        if plot:
            x, y = zip(*Counter(needed_preds).items())
            y = np.array(y, np.float32) / X.shape[0]
            plt.figure()
            plt.grid()
            plt.plot(x, y)
            plt.xticks(np.arange(clf.n_classes_)+1)
            plt.xlabel('Number of predictions needed to gain at least %d%% confident' % (100*k))
            plt.ylabel('Fractions of samples')
            plt.title('%s predicts with mean of %d%% confidence' % (type(clf).__name__, 100*confidence.mean()))
            plt.show()
            print 'Mean number of predictions:', needed_preds.mean()
    else:
        correct = top_predictions[:,:k] == Y[:,None]
        correct = correct.any(axis=1)
    return correct.mean()
