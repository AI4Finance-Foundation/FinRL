:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

Explainable FinRL: An Empirical Approach
===============================





First of all, our work aims to provide an empirical approach to explain the portfolio management task on the basis of FinRL settings. 

We propose an empirical approach to explain the strategies of DRL agents for the portfolio management task:

- First, we study the portfolio management strategy using feature weights, which quantify the relationship between the reward (say, portfolio return) and the input (say, features). In particular, we use the coefficients of a linear model in hindsight as the reference feature weights.

- Then, for the deep reinforcement learning strategy, we use integrated gradients to define the feature weights, which are the coefficients between reward and features under a linear regression model

- Finally, we quantify the prediction power by calculating the linear correlations between the coefficients of a DRL agent and the reference feature weights, and similarly for conventional machine learning methods. Moreover, we consider both the single-step case and multiple-step case.


.. image:: ../image/ExplainableFinRL-ReferenceModel.png


Step 1. Portfolio Management Task
---------------------------------------

Consider a portfolio with 𝑁 risky assets over 𝑇 time slots, the portfolio management task aims to maximize profit and minimize risk.

- The price relative vector :math:`y(𝑡) \in R^N` is defined as the element-wise division of p(𝑡) by p(𝑡-1): y(𝑡) ≜  [p1(t)p1(t-1), p2(t)p2(t-1), … , pN(t)pN(t-1)]T, 𝑡 = 1, ....𝑇 ,  where p(0) ∈ RN is the vector of opening prices at 𝑡 = 1 and  p(t)  ∈ RN denotes the closing prices of all assets at time slot 𝑡 = 1, ...,𝑇 . 

- Let w(0)  ∈ RN denotes the portfolio weights, which is updated at the beginning of time slot 𝑡. 

- The rate of portfolio return is w(t)Ty(t)− 1, and the logarithmic rate of portfolio return is ln(w(t)Ty(t)). 

- The risk of a portfolio is defined as the variance of the rate of portfolio return: w(t)T (t)w(t), where (t) = Cov(y(𝑡)) ∈ RN  N is the covariance matrix of the stock returns at the end of time slot t.

- Our goal is to find a portfolio weight vector w*(t) ∈ RN such that 
w*(t)  ≜argmaxw(𝑡)  w(t)Ty(t) − 𝜆 w(t)T (t)w(t), s.t.  Ni=1 wi (𝑡) = 1, wi (𝑡) ∈ [0, 1], 𝑡 = 1, ..., T, where 𝜆 is the risk aversion parameter


Step 2. The DRL Agent Settings For Portfolio Management Task
---------------------------------------

Similar to the tutorial FinRL: Multiple Stock Trading,  we model the portfolio management process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem. The algorithm is trained using Deep Reinforcement Learning (DRL) algorithms and the components of the reinforcement learning environment are:

- Action: The action space describes the allowed actions an agent can take at a state. In our task, the action w(t)∈ RN  corresponds to the portfolio weight vector decided at the beginning of time slot 𝑡 and should satisfy the constraints: firstly, each element is between 0 and 1, secondly the summation of all elements is 1.

- Reward function: The reward function 𝑟(s(t),w(t),s(t+1)) is the incentive for an agent to learn a profitable policy. We use the logarithmic rate of portfolio return: ln(w(t)Ty(t)).as the reward, where y(t) ∈ RN is the price relative vector.

- State: describes an agent’s perception of a market. The state at the beginning of time slot 𝑡 is s(t) = [f1(𝑡), ..., fK(𝑡), (t)] ∈ RN(N+K)  , 𝑡 = 1, ...,𝑇 .

- DRL Algorithms: We use two popular deep reinforcement learning algorithms: Advantage Actor Critic (A2C)  and Proximal Policy Optimization (PPO).

- Environment: Dow Jones 30 constituent stocks during 01/01/2009 to 09/01/2021
 

DRL Agents

We use integrated gradients to define the feature weights for DRL agents in portfolio management tasks.
                              IG(x)i := (xi - x'i)  1z=0 F(x' + z (x - x'))xidz,
where x  RN is the input and F() is the DRL model. Likewise, we use linear regression coefficients to help understand DRL agents:

wDRL(t)  y(t) = c0(t) [1, ..., 1]T + c1(t)f1(t) + ... + cK(t)fK(t) + (t).

Lastly, we define the feature weights of DRL agents in portfolio management task using integrated gradients and the regression coefficients.
                                                      M(t) := [ M(t)1, ... ,M(t)K ], 
where M(t)k:= Ni=1 IG(fk(t))i Ni=1fk(t)ii=1lE[wDRL(t+l)Ty(t+l) | sk,i(t), w(t)]fk(t)i
                      =  Ni=1fk(t)ii=1lE[ck(t+l)fk(t+l)ifk(t)i | sk,i(t), w(t)]
                      




Step 3. The Feature Weights For Machine Learning Methods
---------------------------------------

We use conventional machine learning methods as comparison. 

- Firstly, it uses the features as input to predict the stock returns vector. 

- Secondly, it builds a linear regression model to find the relationship between the portfolio return vector q and features.

- Lastly, it uses the regression coefficients b to define the feature weights as follows.

We define the feature weights for machine learning methods as 
b(t) := [b(t)1, b(t)2, ..., b(t)K]  RK, where b(t)k = Ni=1  bk(t)fk(t)i ,  bk(t) is the coefficient in the linear model: 
wML(t)  y(t) = b0(t) [1, ..., 1]T + b1(t)f1(t) + ... + bK(t)fK(t) + (t)


Step 4. The Prediction Power
---------------------------------------

Both the machine learning methods and DRL agents take profits from their prediction power. We quantify the prediction power by calculating the linear correlations between the feature weights of a DRL agent and the reference feature weights and similarly for machine learning methods. Furthermore, the machine learning methods and DRL agents are different when predicting the future. The machine learning methods rely on single-step prediction to find portfolio weights. However, the DRL agents find portfolio weights with a long-term goal. Then, we compare two cases, single-step prediction and multi-step prediction.

.. image:: ../image/ExplainableFinRL-ReferenceFeature.png

Step 5. Experiment & Conclusions
---------------------------------------

Our experiment environment is as follows:

Algorithms: PPO, A2C, SVM, Decision Tree, Random Forest, Linear Regression

Data: Dow Jones 30 constituent stocks, accessed at 7/1/2020. We used the data from 1/1/2009 to 6/30/2020 as a training set and the data from 7/1/2020 to 9/1/2021 as a trading set.

We used four technical indicators as features: MACD, CCI, RSI, ADX

Benchmark: Dow Jones Industrial Average (DJIA)

The experiment result shows below:

We firstly compare the portfolio performance among the algorithms

.. image:: ../image/ExplainableFinRL-CumulativeReturn.png


.. image:: ../image/ExplainableFinRL-PerformanceAlgs.png

We find that the DRL methods performed best among all and we seek to explain this empirically using our proposed method.

.. image:: ../image/ExplainableFinRL-SingleStepPrediction.png


