# FinRL Imitation Learning

A multi-stage machine learning approach is a promising methodology for analyzing financial big data, especially when learning from alpha factors or smart investors. Our proposed system automates this workflow, starting with replicating these strategies, and then using reinforcement learning techniques to further refine the results.

By first replicating successful strategies, our system can verify the effectiveness of the approach and identify potential weaknesses or areas for improvement. This process provides a strong foundation for further refinement using reinforcement learning, which can help to optimize and fine-tune the strategies for maximum performance.

By automating this workflow, our system can save valuable time and resources, while also providing more accurate and reliable results. Ultimately, this approach can help to identify profitable investment opportunities and inform smarter financial decision-making.

## **File Structure

**1-Stock Selection**		

We identify a pool of stocks that are favoured by retail investors and have a high correlation between their trading preference and return rates. By analyzing the trading behaviour of retail investors, we can gain valuable insights into the stocks that are popular among this group and understand how their trading preferences affect the performance of these stocks.

**2-Weight Initialization**

We construct our action space, which will serve as our data labels. The action space is a critical component of our machine learning approach, as it represents the set of actions that our algorithm can take in response to the data input. There are two key sources to inform the action construction: MVO (mean-variance optimization) and retail investor preferred weights.  

**3-Imitation Sandbox**

We use a set of regression models, including linear models, trees, and neural networks, to analyze our data. Our approach involves incrementally increasing the complexity of the models to evaluate their performance in predicting outcomes.

To ensure the reliability of our analysis, we conduct a placebo test to evaluate the potential for information leakage. This involves feeding simulated data into our models to assess their performance in predicting outcomes that are not based on actual data. By doing so, we can ensure that our models are not biased by any unforeseen factors or hidden information that may have influenced the results.