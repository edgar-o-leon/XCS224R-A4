#!/usr/bin/env python3
import inspect
import unittest
import argparse
from graderUtil import graded, CourseTestRunner, GradedTestCase
import torch
import torch.nn as nn
import os
import numpy as np

import tensorflow.compat.v1 as tf

from autograde_utils import assert_allclose

# Import submission modules for encoder/decoder testing

from submission.meta_rl.embed.encoder_decoder import EncoderDecoder
from submission.meta_rl.embed.embedders import *
from submission.goal_conditioned_rl.run_episode import run_episode
from submission.goal_conditioned_rl.trainer import update_replay_buffer
from submission.goal_conditioned_rl.utils import HERType

# Import reference solution for encoder/decoder testing
if os.path.exists("./solution"):
    from solution.meta_rl.embed.encoder_decoder import EncoderDecoder as RefEncoderDecoder
    from solution.goal_conditioned_rl.run_episode import run_episode as ref_run_episode


GRADING_RUBRIC_DREAM = {
    'dream/tensorboard/episode': 0.68
}

NUM_BITS = ['6', '15', '25']
HER_TYPES = ['no_hindsight', 'final', 'random', 'future']

GRADING_RUBRIC_HER_SUCCESS_RATE = {
    'bit_flip/num_bits:6/HER_type:no_hindsight': 1.0,
    'bit_flip/num_bits:6/HER_type:final': 1.0,
    'bit_flip/num_bits:15/HER_type:final': 1.0,
    'bit_flip/num_bits:25/HER_type:final': 1.0,
    'bit_flip/num_bits:15/HER_type:random': 1.0,
    'bit_flip/num_bits:15/HER_type:future': 1.0,
}

def safe_parse_file(file, parse_func, default_val = 0):
    try:
        out = parse_func(file)
    except:
        out = default_val
    return out

def parse_her_file(file):
    success_rates = []

    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'eval_metrics/success_rate':
                success_rates.append(v.simple_value)

    if len(success_rates) == 0:
        return 0.0
    else:
        max_success_rate = np.max(np.array(success_rates))

    return max_success_rate

def parse_dream_file(file):
    rewards_test = []

    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'reward/test':
                rewards_test.append(v.simple_value)

    if len(rewards_test) == 0:
        return 0.0
    else:
        max_reward_test = np.max(np.array(rewards_test))

    return max_reward_test

def crawl_folders(rootdir):

    results = {'tests': []}

    her_success_rates, dream_rewards = get_scores(rootdir)

    if len(her_success_rates) >=1:    
        for env_and_method in her_success_rates:
            success_rate = her_success_rates[env_and_method]
            
            if success_rate == 1:
                score = 10.
            else:
                score = 0.

            results['tests'].append({
                'score': score,
                'max_score': 10.,
                'success rate': round(success_rate, 1),
                'name': '{}'.format(env_and_method),
                'output': 'Your max success rate was {}, and the max success rate required for full points is {} for {}.'.format(
                    her_success_rates[env_and_method], 1.0, env_and_method
                ),
            })
    else:
        results['tests'].append({
            'score': 0.,
            'name': 'HER',
            'output': 'No run logs found.'
        })
    
    if len(dream_rewards) >= 1:
        for entry in dream_rewards:
            reward = dream_rewards[entry]
            
            if reward >= 0.69:
                score = 40.
            else:
                score = 40 * (reward / 0.69)

            results['tests'].append({
                'score': round(score, 1),
                'max_score': 40.,
                'test reward': round(reward, 1),
                'name': '{}'.format(entry),
                'output': 'Your max reward was {}, and the max reward rate required for full points is {} for {}.'.format(
                    reward, 0.69, entry
                ),
            })
            
            # breakpoint()
    else:
       results['tests'].append({
            'score': 0.,
            'name': 'DREAM',
            'output': 'No run logs found.'
        }) 

    return results

def get_scores(rootdir):

    success_rate_dict = {}
    rewards_test_dict = {}

    for env in GRADING_RUBRIC_HER_SUCCESS_RATE:
        success_rate_dict[env] = 0
    
    for env in GRADING_RUBRIC_DREAM:
        rewards_test_dict[env] = 0 
   
    for root, _, filelist in os.walk(rootdir):
        # Skip hidden or system folders like __MACOSX
        if '__MACOSX' in root or '/.' in root:
            continue

        for file in filelist:
            if file.startswith('.') or 'MACOSX' in file:
                continue
            
            if 'event' in file:
                dirname = root
                for env in GRADING_RUBRIC_HER_SUCCESS_RATE:
                    for num_bits in NUM_BITS:
                        for her_type in HER_TYPES:
                            if num_bits in dirname and her_type in dirname and num_bits in env and her_type in env:
                                success_rate = safe_parse_file(root + '/' + file, parse_her_file)
                                print(env, success_rate)
                                success_rate_dict[env] = max(success_rate_dict[env], success_rate)
                
                for dream_env in GRADING_RUBRIC_DREAM:
                    if dream_env in dirname:
                        reward_test = safe_parse_file(root + '/' + file, parse_dream_file)
                        print(dream_env, reward_test)
                        rewards_test_dict[dream_env] = max(rewards_test_dict[dream_env], reward_test)

    return success_rate_dict, rewards_test_dict    

#########
# TESTS #
#########

class GoalConditionedRL_Test(GradedTestCase):
    """Test class for run_episode function."""
    def _create_mock_env(self, episode_length=5, rewards=None, done_at_step=None, success_at_step=None):
        """Create a mock environment for testing."""
        if rewards is None:
            rewards = [1.0] * episode_length

        class MockEnv:
            def __init__(self, episode_length, rewards, done_at_step, success_at_step):
                self.episode_length = episode_length
                self.rewards = rewards
                self.done_at_step = done_at_step
                self.success_at_step = success_at_step
                self.step_count = 0
                self.state_dim = 4

            def reset(self):
                self.step_count = 0
                # Return state and goal_state as numpy arrays
                state = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
                goal_state = np.array([0.8, 0.9, 1.0, 1.1], dtype=np.float32)
                return state, goal_state

            def step(self, action):
                # Create next_state based on action
                next_state = np.array([0.1 + action * 0.1, 0.2 + action * 0.1,
                                     0.3 + action * 0.1, 0.4 + action * 0.1], dtype=np.float32)

                reward = self.rewards[min(self.step_count, len(self.rewards) - 1)]

                # Check if episode should end
                done = (self.done_at_step is not None and self.step_count >= self.done_at_step) or \
                       (self.step_count >= self.episode_length - 1)

                # Check if goal is reached
                successful = (self.success_at_step is not None and
                            self.step_count >= self.success_at_step)

                info = {"successful_this_state": successful}

                self.step_count += 1
                return next_state, reward, done, info

        return MockEnv(episode_length, rewards, done_at_step, success_at_step)

    def _create_mock_q_net(self, action_values=None):
        """Create a mock Q-network for testing."""
        class MockQNet:
            def __init__(self, action_values):
                # Default action values favor action 1
                self.action_values = action_values or [0.1, 0.8, 0.3, 0.2]

            def __call__(self, state_batch):
                # state_batch should be shape (batch_size, state_dim)
                batch_size = state_batch.shape[0]
                # Return Q-values for all actions
                return torch.tensor([self.action_values] * batch_size, dtype=torch.float32)

        return MockQNet(action_values)

    def _create_mock_replay_buffer(self, buffer_size=1000, batch_size=32):
        """Create a mock replay buffer for testing."""
        class MockReplayBuffer:
            def __init__(self, buffer_size, batch_size):
                self.buffer_size = buffer_size
                self.batch_size = batch_size
                self.experiences = []

            def add(self, state, action, reward, next_state):
                self.experiences.append({
                    'state': state.copy() if hasattr(state, 'copy') else state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.copy() if hasattr(next_state, 'copy') else next_state
                })

            def __len__(self):
                return len(self.experiences)

            def get_experiences(self):
                return self.experiences

        return MockReplayBuffer(buffer_size, batch_size)

    def _create_mock_episode_experience(self, episode_length=5, state_dim=4):
        """Create mock episode experience data."""
        episode_experience = []
        for t in range(episode_length):
            state = np.array([0.1 + t * 0.1] * state_dim, dtype=np.float32)
            action = t % 4
            reward = -1.0 if t < episode_length - 1 else 0.0  # Sparse reward
            next_state = np.array([0.1 + (t + 1) * 0.1] * state_dim, dtype=np.float32)
            goal_state = np.array([1.0] * state_dim, dtype=np.float32)

            episode_experience.append((state, action, reward, next_state, goal_state))

        return episode_experience

    def _create_mock_reward_function(self):
        """Create a simple mock reward function."""
        def reward_function(state, goal):
            # Simple distance-based reward
            distance = np.linalg.norm(state - goal)
            return 0.0 if distance < 0.1 else -1.0

        return reward_function

class Test_1a(GoalConditionedRL_Test):
    @graded()
    def test_0(self):
        """1a-0-basic: test basic run_episode functionality"""
        env = self._create_mock_env(episode_length=3)
        q_net = self._create_mock_q_net()
        steps_per_episode = 3

        episode_experience, episodic_return, succeeded = run_episode(
            env, q_net, steps_per_episode)

        # Check return types
        self.assertIsInstance(episode_experience, list)
        self.assertIsInstance(episodic_return, float)
        self.assertIsInstance(succeeded, bool)

        # Check episode length
        self.assertGreater(len(episode_experience), 0)
        self.assertLessEqual(len(episode_experience), steps_per_episode)

        # Check experience tuple structure
        for experience in episode_experience:
            self.assertEqual(len(experience), 5)  # (state, action, reward, next_state, goal)
            state, action, reward, next_state, goal_state = experience

            # Check types and shapes
            self.assertIsInstance(state, np.ndarray)
            self.assertIsInstance(action, int)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(next_state, np.ndarray)
            self.assertIsInstance(goal_state, np.ndarray)

            # Check state dimensions
            self.assertEqual(len(state), 4)
            self.assertEqual(len(next_state), 4)
            self.assertEqual(len(goal_state), 4)

    @graded()
    def test_1(self):
        """1a-1-basic: test episodic return calculation"""
        rewards = [1.0, 2.0, 3.0]
        env = self._create_mock_env(episode_length=3, rewards=rewards)
        q_net = self._create_mock_q_net()

        episode_experience, episodic_return, _ = run_episode(
            env, q_net, steps_per_episode=3)

        # Check that episode actually ran (not empty)
        self.assertGreater(len(episode_experience), 0,
                          "Episode experience should not be empty - implement the episode loop")

        # Check that episodic return is sum of rewards
        expected_return = sum(rewards[:len(episode_experience)])
        self.assertAlmostEqual(episodic_return, expected_return, places=5)

    @graded()
    def test_2(self):
        """1a-2-basic: test early episode termination"""
        env = self._create_mock_env(episode_length=10, done_at_step=2)
        q_net = self._create_mock_q_net()

        episode_experience, _, _ = run_episode(
            env, q_net, steps_per_episode=10)

        # Episode should run but terminate early (not be empty)
        self.assertGreater(len(episode_experience), 0,
                          "Episode experience should not be empty - implement the episode loop")
        # Episode should terminate early
        self.assertLessEqual(len(episode_experience), 3)

    @graded()
    def test_3(self):
        """1a-3-basic: test success detection"""
        env = self._create_mock_env(episode_length=5, success_at_step=2)
        q_net = self._create_mock_q_net()

        _, _, succeeded = run_episode(
            env, q_net, steps_per_episode=5)

        # Should detect success
        self.assertTrue(succeeded)

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_1b(GoalConditionedRL_Test):
    @graded()
    def test_0(self):
        """1b-0-basic: test HER FINAL strategy"""
        replay_buffer = self._create_mock_replay_buffer()
        episode_experience = self._create_mock_episode_experience()
        reward_function = self._create_mock_reward_function()

        update_replay_buffer(
            replay_buffer,
            episode_experience,
            her_type=HERType.FINAL,
            env_reward_function=reward_function
        )

        # Should have original experiences + FINAL relabeled experiences
        expected_count = 2 * len(episode_experience)
        self.assertEqual(len(replay_buffer), expected_count)

    @graded()
    def test_1(self):
        """1b-1-basic: test HER RANDOM strategy"""
        replay_buffer = self._create_mock_replay_buffer()
        episode_experience = self._create_mock_episode_experience()
        reward_function = self._create_mock_reward_function()
        num_relabeled = 3

        update_replay_buffer(
            replay_buffer,
            episode_experience,
            her_type=HERType.RANDOM,
            env_reward_function=reward_function,
            num_relabeled=num_relabeled
        )

        # Should have original + num_relabeled per transition
        expected_count = len(episode_experience) * (1 + num_relabeled)
        self.assertEqual(len(replay_buffer), expected_count)

    @graded()
    def test_2(self):
        """1b-2-basic: test HER FUTURE strategy"""
        replay_buffer = self._create_mock_replay_buffer()
        episode_experience = self._create_mock_episode_experience()
        reward_function = self._create_mock_reward_function()
        num_relabeled = 2

        update_replay_buffer(
            replay_buffer,
            episode_experience,
            her_type=HERType.FUTURE,
            env_reward_function=reward_function,
            num_relabeled=num_relabeled
        )

        # Should have original + num_relabeled per transition
        expected_count = len(episode_experience) * (1 + num_relabeled)
        self.assertEqual(len(replay_buffer), expected_count)

class Test_1f(GradedTestCase):
    """Test class for checking HER training logs against GRADING_RUBRIC_HER_SUCCESS_RATE."""
    current_directory = os.getcwd()

    LOGS_DIR = os.path.join(current_directory, 'submission', 'goal_conditioned_rl', 'logs', 'gcrl')

    @graded()
    def test_0(self):
        """1f-0-basic: Check HER training logs exist and meet success rate thresholds"""
        if not os.path.exists(self.LOGS_DIR):
            self.fail(f"Logs directory '{self.LOGS_DIR}' not found. Please include your training logs.")

        success_rate_dict, _ = get_scores(self.LOGS_DIR)

        missing_or_failing = []
        passing = []

        for config, threshold in GRADING_RUBRIC_HER_SUCCESS_RATE.items():
            # Extract num_bits and her_type from config
            parts = config.split('/')
            num_bits = parts[1].split(':')[1]  # e.g., '6'
            her_type = parts[2].split(':')[1]  # e.g., 'final'

            # Find matching entry in success_rate_dict
            found = False
            for env, success_rate in success_rate_dict.items():
                if f'num_bits:{num_bits}' in env and f'HER_type:{her_type}' in env:
                    found = True
                    if success_rate >= threshold:
                        passing.append((config, success_rate))
                    else:
                        missing_or_failing.append((config, success_rate, threshold))
                    break

            if not found:
                missing_or_failing.append((config, 0.0, threshold))

        for config, rate in passing:
            print(f"PASS: {config} with success_rate={rate}")

        for config, rate, threshold in missing_or_failing:
            print(f"FAIL: {config} with success_rate={rate} (required: {threshold})")

        self.assertEqual(
            len(missing_or_failing), 0,
            f"Missing or failing configurations: {[c[0] for c in missing_or_failing]}"
        )


class DREAM_Test(GradedTestCase):
    """Test class for EncoderDecoder functions: label_rewards and _compute_losses."""

    def setUp(self):
        """Set up test fixtures with mock components and data."""
        try:
            torch.manual_seed(224)
            np.random.seed(224)

            # Test dimensions
            self.batch_size = 3
            self.episode_len = 5
            self.embed_dim = 16
            self.penalty = 0.1

            self.device = torch.device("cpu")

            # Use modern PyTorch API for setting defaults
            torch.set_default_dtype(torch.float32)
            torch.set_default_device("cpu")

            # Create mock embedders
            self.transition_embedder = self._create_mock_transition_embedder()
            self.id_embedder = self._create_mock_id_embedder()

            # Move embedders to the correct device
            self.transition_embedder.to(self.device)
            self.id_embedder.to(self.device)

            # Create EncoderDecoder instance
            self.encoder_decoder = EncoderDecoder(
                self.transition_embedder,
                self.id_embedder,
                self.penalty,
                self.embed_dim
            )
            self.encoder_decoder.to(self.device)
        except Exception as e:
            error_msg = f"setUp failed with exception: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise Exception(error_msg) from e

    def _create_mock_transition_embedder(self):
        """Create a mock transition embedder for testing."""
        class MockTransitionEmbedder(nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.embed_dim = embed_dim
                self.linear = nn.Linear(10, embed_dim)  # Arbitrary input size

            def forward(self, experiences):
                # Mock embedding based on experience index - return deterministic values
                batch_size = len(experiences)

                # Check if we're on CUDA and get the correct device
                device = next(self.parameters()).device

                # Create deterministic input based on experience properties
                mock_input = torch.zeros(batch_size, 10, device=device)
                for i, exp in enumerate(experiences):
                    # Use some properties of the experience to create deterministic input
                    mock_input[i, 0] = float(exp.action % 4)
                    mock_input[i, 1] = float(exp.reward)
                    mock_input[i, 2] = float(exp.state.env_id)
                    # Fill remaining with small random values based on index
                    mock_input[i, 3:] = torch.sin(torch.arange(7, device=device) * (i + 1) * 0.1)

                return self.linear(mock_input)

        return MockTransitionEmbedder(self.embed_dim)

    def _create_mock_id_embedder(self):
        """Create a mock ID embedder for testing."""
        class MockIDEmbedder(nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.embed_dim = embed_dim
                self.embedding = nn.Embedding(10, embed_dim)  # Support 10 different IDs

            def forward(self, env_ids):
                if isinstance(env_ids, list):
                    device = next(self.parameters()).device
                    env_ids = torch.tensor(env_ids, device=device)
                return self.embedding(env_ids)

        return MockIDEmbedder(self.embed_dim)

    def _create_mock_trajectories(self, batch_size=None, episode_lengths=None):
        """Create mock trajectory data for testing."""
        if batch_size is None:
            batch_size = self.batch_size
        if episode_lengths is None:
            episode_lengths = [self.episode_len] * batch_size

        class MockState:
            def __init__(self, env_id):
                self.env_id = env_id

        class MockExperience:
            def __init__(self, state, action=None, reward=None):
                self.state = state
                self.action = action if action is not None else np.random.randint(0, 4)
                self.reward = reward if reward is not None else np.random.random()

        trajectories = []
        for b in range(batch_size):
            trajectory = []
            env_id = b % 3  # Cycle through environment IDs
            for t in range(episode_lengths[b]):
                state = MockState(env_id)
                experience = MockExperience(state)
                trajectory.append(experience)
            trajectories.append(trajectory)

        return trajectories


class Test_2c(DREAM_Test):
    @graded()
    def test_0(self):
        """2c-0-basic: test _compute_losses shape and basic functionality"""
        trajectories = self._create_mock_trajectories()

        # Get embeddings from _compute_embeddings (this is provided to students)
        id_embeddings, all_decoder_embeddings, decoder_embeddings, mask = (
            self.encoder_decoder._compute_embeddings(trajectories))

        # Test _compute_losses function
        losses = self.encoder_decoder._compute_losses(
            trajectories, id_embeddings, all_decoder_embeddings, decoder_embeddings, mask)
        # Check that losses dict contains expected keys
        self.assertIn("decoder_loss", losses)
        self.assertIn("information_bottleneck", losses)

        # Check that losses are scalar tensors
        self.assertEqual(losses["decoder_loss"].dim(), 0)
        self.assertEqual(losses["information_bottleneck"].dim(), 0)

        # Check that losses are finite and non-negative
        self.assertTrue(torch.isfinite(losses["decoder_loss"]))
        self.assertTrue(torch.isfinite(losses["information_bottleneck"]))
        self.assertGreaterEqual(losses["decoder_loss"].item(), 0)
        self.assertGreaterEqual(losses["information_bottleneck"].item(), 0)

        # Test gradient flow - decoder_loss should have gradients w.r.t decoder params
        # but not w.r.t id_embeddings (due to detach)
        losses["decoder_loss"].backward(retain_graph=True)

        # Check that some parameters have gradients
        has_grad = any(p.grad is not None for p in self.encoder_decoder.parameters())
        self.assertTrue(has_grad, "No gradients found for decoder parameters")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_2d(DREAM_Test):
    @graded()
    def test_0(self):
        """2d-0-basic: test label_rewards shape and basic functionality"""
        trajectories = self._create_mock_trajectories()

        rewards, distances = self.encoder_decoder.label_rewards(trajectories)
        
        # Check shapes
        expected_reward_shape = (self.batch_size, self.episode_len)
        expected_distance_shape = (self.batch_size, self.episode_len + 1)

        self.assertEqual(rewards.shape, expected_reward_shape,
                        f"Expected rewards shape {expected_reward_shape}, got {rewards.shape}")
        self.assertEqual(distances.shape, expected_distance_shape,
                        f"Expected distances shape {expected_distance_shape}, got {distances.shape}")

        # Check that outputs are detached (no gradients)
        self.assertFalse(rewards.requires_grad, "Rewards should be detached from computation graph")

        # Check that distances are non-negative (since they represent -log probabilities)
        self.assertTrue(torch.all(distances >= 0), "Distances should be non-negative")

        # Check finite values
        self.assertTrue(torch.all(torch.isfinite(rewards)), "Rewards should be finite")
        self.assertTrue(torch.all(torch.isfinite(distances)), "Distances should be finite")

    ### BEGIN_HIDE ###
    ### END_HIDE ###

    @graded()
    def test_2(self):
        """2d-2-basic: test integration with varying trajectory lengths"""
        # Test with different episode lengths to verify masking works correctly
        episode_lengths = [3, 5, 2]  # Different lengths for each batch element
        trajectories = self._create_mock_trajectories(
            batch_size=3, episode_lengths=episode_lengths)

        # Test _compute_losses
        id_embeddings, all_decoder_embeddings, decoder_embeddings, mask = (
            self.encoder_decoder._compute_embeddings(trajectories))
        losses = self.encoder_decoder._compute_losses(
            trajectories, id_embeddings, all_decoder_embeddings, decoder_embeddings, mask)

        # Should still produce valid scalar losses
        self.assertEqual(losses["decoder_loss"].dim(), 0)
        self.assertTrue(torch.isfinite(losses["decoder_loss"]))

        # Test label_rewards
        rewards, distances = self.encoder_decoder.label_rewards(trajectories)

        # Check shapes - should be padded to max length
        max_len = max(episode_lengths)
        expected_reward_shape = (3, max_len)
        expected_distance_shape = (3, max_len + 1)

        self.assertEqual(rewards.shape, expected_reward_shape)
        self.assertEqual(distances.shape, expected_distance_shape)

        # Verify masking - padded positions should have zero rewards
        for i, length in enumerate(episode_lengths):
            if length < max_len:
                # Check that padded rewards are zero
                padded_rewards = rewards[i, length:]
                self.assertTrue(torch.allclose(padded_rewards, torch.zeros_like(padded_rewards)),
                              f"Padded rewards should be zero for trajectory {i}")
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_2e(GradedTestCase):
    """Test class for checking DREAM training logs against GRADING_RUBRIC_DREAM."""

    # Required DREAM configurations and their reward thresholds
    REQUIRED_CONFIGS = GRADING_RUBRIC_DREAM

    current_directory = os.getcwd()

    LOGS_DIR = os.path.join(current_directory, 'submission', 'meta_rl', 'experiments')

    @graded()
    def test_0(self):
        """2e-0-basic: Check DREAM training logs exist and meet reward thresholds"""
        if not os.path.exists(self.LOGS_DIR):
            self.fail(f"Logs directory '{self.LOGS_DIR}' not found. Please include your training logs.")

        _, rewards_test_dict = get_scores(self.LOGS_DIR)

        missing_or_failing = []
        passing = []

        for config, threshold in self.REQUIRED_CONFIGS.items():
            # Find matching entry in rewards_test_dict
            found = False
            for env, reward in rewards_test_dict.items():
                if config in env or env in config:
                    found = True
                    if reward >= threshold:
                        passing.append((config, reward))
                    else:
                        missing_or_failing.append((config, reward, threshold))
                    break

            if not found:
                # Check if any reward was recorded for this config
                reward = rewards_test_dict.get(config, 0.0)
                if reward >= threshold:
                    passing.append((config, reward))
                else:
                    missing_or_failing.append((config, reward, threshold))

        for config, reward in passing:
            print(f"PASS: {config} with max_reward={reward:.3f}")

        for config, reward, threshold in missing_or_failing:
            print(f"FAIL: {config} with max_reward={reward:.3f} (required: {threshold})")

        self.assertEqual(
            len(missing_or_failing), 0,
            f"Missing or failing configurations: {[c[0] for c in missing_or_failing]}"
        )


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test or mode
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test_case",
        nargs="?",
        default="all",
        help="Use 'all' (default), a specific test id like '1-3-basic', 'public', or 'hidden'",
    )
    test_id = parser.parse_args().test_case

    def _flatten(suite):
        """Recursively flatten unittest suites into individual tests."""
        for x in suite:
            if isinstance(x, unittest.TestSuite):
                yield from _flatten(x)
            else:
                yield x

    assignment = unittest.TestSuite()

    if test_id not in {"all", "public", "hidden"}:
        # Run a single specific test
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        # Discover all tests
        discovered = unittest.defaultTestLoader.discover(".", pattern="grader.py")

        if test_id == "all":
            assignment.addTests(discovered)
        else:
            # Filter tests by visibility marker in docstring ("basic" for public tests, "hidden" for hidden tests)
            keyword = "basic" if test_id == "public" else "hidden"
            filtered = [
                t for t in _flatten(discovered)
                if keyword in (getattr(t, "_testMethodDoc", "") or "")
            ]
            assignment.addTests(filtered)

    CourseTestRunner().run(assignment)
