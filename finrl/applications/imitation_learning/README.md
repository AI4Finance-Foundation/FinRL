# FinRL Imitation Learning

A multi-stage machine learning approach is a promising approach for analyzing financial big data, especially when learning from alpha factors or smart investors. Here, we automate this workflow, starting with imitating these strategies, and then using reinforcement learning method to further refine the results.

In complicated tasks such as Go and Atari games, imitation learning is often used to initialize deep neural networks that achieve human-level performance. Imitation learning involves training a model to imitate a human's behavior, typically using a dataset of expert demonstrations. This process provides a a starting point for further refinement using reinforcement learning, which could learn through trial and error to find strategies that surpass human-level performance.

By automating this workflow, we can analogously save valuable time and resources, while also providing more accurate and reliable results. Ultimately, this approach can help to identify profitable investment opportunities and inform smarter financial decision-making.

## File Structure

**1-Stock Selection**		

We identify a pool of stocks that are favoured by retail investors and have a high correlation between their trading preference and return rates. By analyzing the trading behaviour of retail investors, we can gain valuable insights into the stocks that are popular among this group and understand how their trading preferences affect the performance of these stocks.

**2-Weight Initialization**

We construct the action space, which will serve as data labels. The action space is a critical component of a machine learning approach, as it represents the set of actions that our algorithm can take in response to a data input. There are two key sources to inform the action construction: MVO (mean-variance optimization) and retail investor preferred weights.  

**3-Imitation Sandbox**

We use a set of regression models, including linear models, trees, and neural networks, to analyze our data. Our approach involves incrementally increasing the complexity of the models to evaluate their performance in predicting outcomes.

To ensure the reliability of our analysis, we conduct a placebo test to evaluate the potential for information leakage. This involves feeding simulated data into our models to assess their performance in predicting outcomes that are not based on actual data. By doing so, we can ensure that our models are not biased by any unforeseen factors or hidden information that may have influenced the results.
