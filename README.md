Regression Case Study
======================

The goal is to predict the sale price of a particular piece of
heavy equipment at auction based on it's usage, equipment type, and
configuration.  The data is sourced from auction result postings and includes
information on usage and equipment configurations.

Evaluation
======================
We evaluated our model based on Root Mean Squared Log Error.
Which is computed as follows:

![Root Mean Squared Logarithmic Error](images/rmsle.png)

where *p<sub>i</sub>* are the predicted values and *a<sub>i</sub>* are the
target values.

Note that this loss function is sensitive to the *ratio* of predicted values to
the actual values, a prediction of 200 for an actual value of 100 contributes
approximately the same amount to the loss as a prediction of 2000 for an actual
value of 1000.

This loss function is implemented in score_model.py.

Data
======================
The data for this case study are in `./data`. 

