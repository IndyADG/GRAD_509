from sklearn.linear_model import LogisticRegression

# 2D array for Weekly Usage Hrs samples
X = [[1], [3], [5], [7], [9]]
# 1D array for Renewal Status targets/labels
y = [0, 0, 1, 1, 1]

# Q1: Use logistic regression to model the probability of renewal based on weekly usage hours.
print("# Q1: Use logistic regression to model the probability of renewal based on weekly usage hours:")
model = LogisticRegression()
model.fit(X, y)
print("> w0 Bias/Intercept:", model.intercept_[0])
print("> w1 Coefficient:", model.coef_[0][0])

print("Therefore, P(renewal = 1 | x) can be modeled as: sigma(z) = 1/1+e^-z, where " +
      " z = " + str(f"{model.intercept_[0]:,.7f}") + " + " + str(f"{model.coef_[0][0]:,.7f}") + "x.")

# Q2: What is the probability that a user logging 4 hours per week will renew their subscription?
print("\n# Q2: What is the probability that a user logging 4 hours per week will renew their subscription?")
expected_renewal_prob = model.predict_proba([[4]])[0][1]
print("Using the fitted model above, P(renewal = 1 | wkly_usage = 4) is ~" +
      str(f"{expected_renewal_prob:,.7f}")+ ", or about " + str(f"{expected_renewal_prob*100:.1f}%")+ ".")
print("(Note: To retrieve the probability of renewal, we select the second value in the array returned from " +
      "\nmodel.predict_proba(), since the positive class, 1, is returned second " +
      "in model.classes_," + str(model.classes_) + ".)")

# Q3: Based on the model, recommend a threshold for targeting at-risk users with promotions.
print("\n# Q3: Based on the model, recommend a threshold for targeting at-risk users with promotions:")
print("Given that Q2 showed users logging 4 hrs/wk are predicted to be nearly evenly split 50:50 renewing " +
      "\ntheir subscription, a threshold of P(renewal = 1) < 0.5, corresponding roughly to that weekly usage of " +
      "\n< 4 hours, would be a reasonable promotion threshold to consider; any user falling below 4 hours weekly "
      "\nusage is less likely to renew and can be targeted with promotions." +
      "\n\nIf the promotions are expensive, however, we could instead consider narrowing the range of targeted" +
      "\nat-risk users to those more likely to be converted, say only those users with between 3.5 to 4 hrs " +
      "\nweekly usage, which the model predicts to have a P(renewal = 1) between " +
      str(f"{model.predict_proba([[3.5]])[0][1]*100:,.1f}%") + " and " +
      str(f"{model.predict_proba([[4]])[0][1]*100:,.1f}%") + ", pre-promotions.")