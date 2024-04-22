---
layout: post
title: Calculating Statistical Power When Your Analysis Requires the Delta Method
tags:
- Statistics
- A/B Testing
- Practical
summary: Demonstration of a statistical power calculation when the variable in question requires the use of the delta method-- i.e., it is a ratio metric.
---

In this post, we explore a situation typical of a website, where a user is allowed to view the page multiple times and may click on a button of interest on any visit. We would like to perform an A/B test on the non-user level metric, the click-through rate (CTR), defined via $$\text{total clicks}/\text{total page views}$$. In order to do the final analysis, we would need to estimate the variance for the variable via the delta method. However, what does that mean for a priori statistical power calculations?

In my last post, we discussed the linearized form of the quotient $$\frac{\bar{V}}{\bar{M}} =\frac{\bar{V}}{\mu_{M}} - \frac{\mu_{V}}{\mu_{M}^2}\bar{M}.$$ We want to connect how to use this formula to derive a power calculation when using the delta method. This approach is useful for ratio metrics, and many session / page / non-user level metrics like the click-through rate.


## Data Generation

To simulate the problem, we create users that sample a user level click-through rate (CTR) from a normal distribution with mean equal to a given population CTR and a set standard error. For users in the treatment group, we add a small effect to their user-level CTR. We also assign the number of sessions per user that the user will appear in the experiment if they are selected. From there, we sample a set number of users and calculate the clicks and page views per users and return that as a dataframe.

```python
import numpy as np
import pandas as pd
from scipy.special import ndtr
POP_CTR = .65
N_USERS = 50000 # total pool of users
SESSIONS_PER_USER = 8
EFFECT =.01
STE = .003

treatment = np.random.choice([0,1], p=[.5,.5], size=N_USERS)
user_rates = np.random.normal(POP_CTR+treatment*EFFECT, STE, N_USERS)
user_session_rates = np.random.normal(SESSIONS_PER_USER, 2, N_USERS)

user_rates[user_rates<0] = 0
user_rates[user_rates>1] = 1

def generate_exp_data(n_users):
    clicks = []
    users = []
    treatments = []
    j=0
    users_in_exp = np.random.choice(N_USERS, n_users) # number of users in the experiment
    while j < n_users:
        k = users_in_exp[j]
        treatment_data = treatment[k]
        user_session_rate = int(np.maximum(1,user_session_rates[k]))
        user_sessions = np.random.randint(np.maximum(user_session_rate-2,1),user_session_rate+2)
        
        click_data = np.random.binomial(1,user_rates[k],size=user_sessions).tolist()
        clicks.extend(click_data)
        treatments.extend([treatment_data]*user_sessions)
        users.extend([k]*user_sessions)
        j+=1
    d = pd.DataFrame({'click': clicks, 'user' : users, 'treatment':treatments})

    # Linearize for delta method
    d['clicks_user']=d.groupby(['treatment']).click.transform('sum') / d.groupby(['treatment']).user.transform('nunique')
    d['sessions_user']=d.groupby(['treatment']).click.transform('count')/ d.groupby(['treatment']).user.transform('nunique')
    d['session'] = 1
    
    df=d.groupby(['user','treatment'], as_index=False).agg({
        'click' : 'sum',
        'session' : 'sum',
        'clicks_user' : 'max',
        'sessions_user' : 'max'
    })

    # Construct the linearized term for the delta method, which is at the user level!
    df['linear_ctr'] = (1/df.sessions_user)*df.click-(df.clicks_user/df.sessions_user**2)*df.session

    
    return df

def calculate_p_value(df):
    # delta method! linearize the term clicks / sessions    
    diff = df[df.treatment==1].click.sum() / df[df.treatment==1].session.sum() \
        - df[df.treatment==0].click.sum() / df[df.treatment==0].session.sum() 
    var_a = df[df.treatment == 0].linear_ctr.var() / df[df.treatment == 0].user.nunique()
    var_b = df[df.treatment == 1].linear_ctr.var() / df[df.treatment == 1].user.nunique()
    ste = np.sqrt(var_b + var_a)
    p = ndtr(diff / ste)
    return diff, ste, 1-p if p > .5 else p
```

## Calculate $$n$$ from the power formula

To avoid too much complication, we will use [Lehr's rule](https://en.wikipedia.org/wiki/Power_of_a_test) from the wiki where the sample size necessary is 

$$
\begin{equation}
n \approx 16 \frac{\sigma^2}{\delta^2}.
\end{equation}
$$


It may not be obvious, but you can actually linearize your variable of interest, calculate the standard deviation, and plug it into the power formula! Let's do a simulation to convince ourselves. We will calculate an estimated power using the standard formula for $$\alpha=.05
$$ and power at 80%, and then we will simulate to estimate the power.


```python
np.random.seed(777)
d = generate_exp_data(1000) # using a smaller sample than we'd eventually need
sigma = d.linear_ctr.std() # calculate the STD of the linearized term

n = 16 * (sigma**2 / (EFFECT)**2 )
std_err = sigma / np.sqrt(1000)
n # formula per treatment
```




    5289.1898124023655



## Simulate the true power for the given $$n$$

Recall, the power of the test is equivalent to the following probability:

$$\begin{equation}P(\text{null hypothesis rejected} | \text{alternative hypothesis is true}).\end{equation}$$ 

Above, we have constructed a simulation where the alternative hypothesis is true ($$\delta>0$$), and so if we calculate how often we would reject the null hypothesis when $$p < \text{significance level}/2=.05/2$$ with the given sample size derived above will estimate the statistical power.


```python
ps = []
significance_level = .05
for k in range(1000):
    ps.append(calculate_p_value(generate_exp_data(int(2*n)))) # Generate P distribution

```


```python
# power for 2 tailed test, p(H_0 rejected | H_1 is true)
np.mean([p[2] < significance_level/2 for p in ps])
```




    0.809



Voila!
