import matplotlib.pyplot as plt
import numpy as np

from smt.surrogate_models import KRG

# defining the toy example
def target_fun(x):
    import numpy as np

    return np.cos(5 * x)

nobs = 50  # number of obsertvations
xt = np.random.uniform(size=nobs)  # design points

# adding a random noise to observations
yt = target_fun(xt) + np.random.normal(scale=0.05, size=nobs)

# training the model with the option eval_noise= True
sm = KRG(eval_noise=True, hyper_opt="Cobyla")
sm.set_training_values(xt, yt)
sm.train()

# predictions
x = np.linspace(0, 1, 100).reshape(-1, 1)
y = sm.predict_values(x)  # predictive mean
var = sm.predict_variances(x)  # predictive variance

# plotting predictions +- 3 std confidence intervals
plt.rcParams["figure.figsize"] = [4, 3]
plt.fill_between(
    np.ravel(x),
    np.ravel(y - 3 * np.sqrt(var)),
    np.ravel(y + 3 * np.sqrt(var)),
    alpha=0.2,
    label="Confidence Interval 99%",
)
plt.scatter(xt, yt, label="Training noisy data", s=70)
plt.plot(x, y, label="Prediction", linewidth=4, color='tab:blue')
# plt.plot(x, target_fun(x), label="target function", linewidth=3)
# plt.title("Kriging model with noisy observations")
# plt.legend(loc=0)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$y$")

# turn off plot axes
plt.gca().axis('off')
plt.savefig('surrogate.png', bbox_inches='tight', dpi=300, transparent=True)
plt.show()