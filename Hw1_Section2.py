from sklearn.linear_model import LogisticRegression

# 2D array for Weekly Usage Hrs samples
X = [[1], [3], [5], [7], [9]]
# 1D array for Renewal Status targets/labels
y = [0, 0, 1, 1, 1]

# Q1: Use logistic regression to model the probability of renewal based on weekly usage hours.
print("Q1: Use logistic regression to model the probability of renewal based on weekly usage hours:")
model = LogisticRegression()
model.fit(X, y)
print("> w0 Intercept:", model.intercept_[0])
print("> w1 Coefficient:", model.coef_[0][0])

print("Therefore, this Logistic Regression Model is sigmoid(z), where z = " + str(f"{model.intercept_[0]:,.8f}") +
      " + " + str(f"{model.coef_[0][0]:,.8f}") + "x")

# Q2: What is the probability that a user logging 4 hours per week will renew their subscription?
print("# Q2: What is the probability that a user logging 4 hours per week will renew their subscription?")
expected_renewal_prob = model.predict_proba([[4]])[0][1]
print("The predicted probability that a user logging 4 hours per week will renew is " +
      str(f"{expected_renewal_prob:,.3f}")+ ", or " + str(f"{expected_renewal_prob*100:.1f}%")+ ".")
print("Note: To get the probability of renewal, we select the second value in the array returned from " +
      "\n model.predict_proba(), since class label 1 represents renewal and that class is returned second " +
      "from model.classes_, i.e. " + str(model.classes_) + ", here.")

# Q3: Based on the model, recommend a threshold for targeting at-risk users with promotions.