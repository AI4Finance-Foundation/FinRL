The three layers of FinRL library are **stock market environment (FinRL-Meta)**, **DRL trading agent**, and **stock trading applications**. The lower layer provides APIs for the upper layer, making the lower layer transparent to the upper layer. The agent layer interacts with the environment layer in an exploration-exploitation manner, whether to repeat prior working-well decisions or to make new actions hoping to get greater rewards. 

.. image:: ../image/finrl_framework.png
    :width: 80%
    :align: center
