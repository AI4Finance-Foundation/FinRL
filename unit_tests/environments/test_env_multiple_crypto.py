from __future__ import annotations

import unittest

import numpy as np

from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv


def make_env(gamma=0.99, price_change=10.0, buy_cost_pct=0.0):
    prices = np.array(
        [[100.0], [100.0], [100.0 + price_change], [100.0 + 2 * price_change], [100.0]]
    )
    return CryptoEnv(
        config={"price_array": prices, "tech_array": np.zeros_like(prices)},
        initial_capital=1000.0,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=0.0,
        gamma=gamma,
    )


def rollout(env, buy=False):
    rewards = []
    done = False
    while not done:
        # The normalizer maps 0.01 to one unit at the initial price of 100.
        action = 0.01 if buy and env.time == 0 else 0.0
        _, reward, done, _ = env.step(np.array([action]))
        rewards.append(reward)
    return rewards


class TestCryptoEnvReset(unittest.TestCase):
    def test_reset_clears_episode_statistics(self):
        env = make_env()
        env.reset()
        rollout(env, buy=True)
        self.assertNotEqual(env.gamma_return, 0.0)
        self.assertNotEqual(env.episode_return, 0.0)

        state = env.reset()

        self.assertEqual(env.gamma_return, 0.0)
        self.assertEqual(env.episode_return, 0.0)
        np.testing.assert_array_equal(state, make_env().reset())

    def test_idle_episode_does_not_inherit_previous_gains_or_losses(self):
        for gamma in (0.5, 0.99, 1.0):
            for price_change in (-10.0, 10.0):
                with self.subTest(gamma=gamma, price_change=price_change):
                    env = make_env(gamma=gamma, price_change=price_change)
                    env.reset()
                    previous_rewards = rollout(env, buy=True)
                    self.assertNotEqual(previous_rewards[-1], 0.0)

                    env.reset()

                    self.assertEqual(rollout(env), [0.0] * env.max_step)
                    self.assertEqual(env.episode_return, 1.0)

    def test_repeated_episode_matches_fresh_environment(self):
        env = make_env(buy_cost_pct=0.01)
        env.reset()
        expected_rewards = rollout(env, buy=True)

        env.reset()
        actual_rewards = rollout(env, buy=True)

        np.testing.assert_allclose(actual_rewards, expected_rewards)

    def test_reset_during_episode_discards_accumulated_reward(self):
        env = make_env(buy_cost_pct=0.01)
        env.reset()
        _, reward, done, _ = env.step(np.array([0.01]))
        self.assertFalse(done)
        self.assertNotEqual(reward, 0.0)

        env.reset()

        self.assertEqual(rollout(env), [0.0] * env.max_step)


if __name__ == "__main__":
    unittest.main()
