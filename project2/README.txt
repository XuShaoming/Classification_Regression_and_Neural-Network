About report

In the report, we do not use the lambda value and hidden layers number recommended by the assignment document. Instead we use a much wider range of lambdas and hidden units to test our model. I think it will give us more insight about the model.

you can see in our report the lambda values are range from 1 to 100. And the hidden units are range from 0 to 240. You can see all the data in the report/results.xlsx file.

We get a lot of data in test, however, considering to make the concise report we only choose cases with 5, 15, 35, 55 hidden layer to study the relation between lambda and accuracy. And within this range, we find the the 55 units of hidden layers with 30 lambda value give us the highest test accuracy which is 94.51%.

So, if you use your model to test our code, you may need to adjust your test code and let the units of hidden layers to be 55 and lambda to be 30 to get the similar answer as us.  

In deepnnScrip.py, for the same reason, we expand the test range. In our report, the numbers hidden layer are range from 1 to 16 which is much larger than the value recommended by the assignment description.


