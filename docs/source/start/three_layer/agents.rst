:github_url: https://github.com/AI4Finance-Foundation/FinRL

2. DRL Agents
============================

FinRL contains fine-tuned standard DRL algorithms in ElegantRL, Stable Baseline 3, and RLlib. ElegantRL is a scalable and elastic DRL library that maintained by AI4Finance, with faster and more stable performance than Stable Baseline 3 and RLlib. In the *Three-Layer Architecture* section, there will be detailed explanation about how ElegantRL accomplish its role in FinRL perfectly. If interested, please refer to ElegantRL's `GitHub page <https://github.com/AI4Finance-Foundation/ElegantRL>`_ or `documentation <https://elegantrl.readthedocs.io>`_.

With those three powerful DRL libraries, FinRL provides the following algorithms for users:

.. image:: ../image/alg_compare.png

As mentioned in the introduction, FinRL's DRL agents are built by fine-tuned standard DRL algorithms depending on three famous DRL library: ElegantRL, Stable Baseline 3, and RLlib.

The supported algorithms include: DQN, DDPG, Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to design their own DRL algorithms by adapting these DRL algorithms, e.g., Adaptive DDPG, or employing ensemble methods. The comparison of DRL algorithms is shown in the table bellow:

.. image:: ../../image/alg_compare.png
   :align: center

Users are able to choose their favorite DRL agents for training. Different DRL agents might have different performance in various tasks.

ElegantRL: DRL library
------------------------

.. image:: ../../image/ElegantRL_icon.jpeg
    :width: 30%
    :align: center
    :target: https://github.com/AI4Finance-Foundation/ElegantRL


One sentence summary of reinforcement learning (RL): in RL, an agent learns by continuously interacting with an unknown environment, in a trial-and-error manner, making sequential decisions under uncertainty and achieving a balance between exploration (new territory) and exploitation (using knowledge learned from experiences).

Deep reinforcement learning (DRL) has great potential to solve real-world problems that are challenging to humans, such as gaming, natural language processing (NLP), self-driving cars, and financial trading. Starting from the success of AlphaGo, various DRL algorithms and applications are emerging in a disruptive manner. The ElegantRL library enables researchers and practitioners to pipeline the disruptive “design, development and deployment” of DRL technology.

The library to be presented is featured with “elegant” in the following aspects:

    - Lightweight: core codes have less than 1,000 lines, e.g., helloworld.
    - Efficient: the performance is comparable with Ray RLlib.
    - Stable: more stable than Stable Baseline 3.

ElegantRL supports state-of-the-art DRL algorithms, including discrete and continuous ones, and provides user-friendly tutorials in Jupyter notebooks. The ElegantRL implements DRL algorithms under the Actor-Critic framework, where an Agent (a.k.a, a DRL algorithm) consists of an Actor network and a Critic network. Due to the completeness and simplicity of code structure, users are able to easily customize their own agents.

Please refer to ElegantRL's `GitHub page <https://github.com/AI4Finance-Foundation/ElegantRL>`_ or `documentation <https://elegantrl.readthedocs.io>`_ for more details.
