"""
Fraud Detection Model using Reinforcement Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from gymnasium import spaces


class FraudDetectionEnv(gym.Env):
    """
    Custom RL environment for fraud detection in financial transactions
    """

    def __init__(self, transaction_data: pd.DataFrame):
        super(FraudDetectionEnv, self).__init__()

        self.transaction_data = transaction_data
        self.current_step = 0
        self.max_steps = len(transaction_data)

        # Action space: 0 = no fraud, 1 = fraud
        self.action_space = spaces.Discrete(2)

        # Observation space: transaction features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(transaction_data.shape[1] - 1,),  # Exclude target column
            dtype=np.float32
        )

        # Performance tracking
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= self.max_steps:
            raise ValueError("Episode has ended")

        # Get current transaction
        current_transaction = self.transaction_data.iloc[self.current_step]
        true_label = current_transaction['is_fraud']

        # Calculate reward based on action and true label
        reward = self._calculate_reward(action, true_label)

        # Update performance metrics
        self._update_metrics(action, true_label)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Get next observation
        observation = self._get_observation() if not done else np.zeros(self.observation_space.shape)

        info = self._get_info()

        return observation, reward, done, False, info

    def _get_observation(self):
        """Get current observation (transaction features)"""
        if self.current_step >= self.max_steps:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        transaction = self.transaction_data.iloc[self.current_step]
        # Exclude the target column
        features = transaction.drop('is_fraud').values.astype(np.float32)
        return features

    def _calculate_reward(self, action, true_label):
        """Calculate reward based on action and true label"""
        # Reward structure:
        # True Positive (detect fraud correctly): +10
        # True Negative (correctly identify legitimate): +1
        # False Positive (incorrectly flag legitimate): -5
        # False Negative (miss fraud): -20

        if action == 1 and true_label == 1:  # True Positive
            return 10
        elif action == 0 and true_label == 0:  # True Negative
            return 1
        elif action == 1 and true_label == 0:  # False Positive
            return -5
        else:  # False Negative
            return -20

    def _update_metrics(self, action, true_label):
        """Update performance metrics"""
        if action == 1 and true_label == 1:
            self.true_positives += 1
        elif action == 0 and true_label == 0:
            self.true_negatives += 1
        elif action == 1 and true_label == 0:
            self.false_positives += 1
        else:
            self.false_negatives += 1

    def _get_info(self):
        """Get additional information"""
        total_predictions = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives

        if total_predictions == 0:
            return {}

        precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives
        }


class FraudDetectionModel:
    """
    Fraud Detection Model using Reinforcement Learning
    """

    def __init__(self, model_type: str = "PPO"):
        self.model_type = model_type
        self.model = None
        self.env = None

    def prepare_data(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare transaction data for training"""
        # Feature engineering
        prepared_df = transactions_df.copy()

        # Add derived features
        if 'amount' in prepared_df.columns:
            prepared_df['log_amount'] = np.log1p(prepared_df['amount'])
            prepared_df['amount_zscore'] = (prepared_df['amount'] - prepared_df['amount'].mean()) / prepared_df['amount'].std()

        if 'timestamp' in prepared_df.columns:
            prepared_df['timestamp'] = pd.to_datetime(prepared_df['timestamp'])
            prepared_df['hour'] = prepared_df['timestamp'].dt.hour
            prepared_df['day_of_week'] = prepared_df['timestamp'].dt.dayofweek
            prepared_df['is_weekend'] = prepared_df['day_of_week'].isin([5, 6]).astype(int)

        # Handle categorical variables
        categorical_columns = prepared_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'is_fraud':
                prepared_df[col] = pd.Categorical(prepared_df[col]).codes

        # Fill missing values
        prepared_df = prepared_df.fillna(0)

        return prepared_df

    def create_environment(self, transaction_data: pd.DataFrame):
        """Create the RL environment"""
        self.env = FraudDetectionEnv(transaction_data)
        return self.env

    def train(self, transaction_data: pd.DataFrame, total_timesteps: int = 100000):
        """Train the fraud detection model"""
        # Prepare data
        prepared_data = self.prepare_data(transaction_data)

        # Create environment
        env = self.create_environment(prepared_data)

        # Initialize model
        if self.model_type == "PPO":
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5
            )
        elif self.model_type == "DQN":
            self.model = DQN(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=1e-4,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05
            )

        # Train the model
        self.model.learn(total_timesteps=total_timesteps)

        return self.model

    def predict(self, transaction_features: np.ndarray) -> Tuple[int, float]:
        """Predict fraud probability for a transaction"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        action, _ = self.model.predict(transaction_features, deterministic=True)
        return int(action), 0.0  # Note: Basic RL models don't provide probabilities

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        prepared_test_data = self.prepare_data(test_data)
        test_env = FraudDetectionEnv(prepared_test_data)

        obs, _ = test_env.reset()
        total_reward = 0
        predictions = []
        true_labels = []

        for i in range(len(prepared_test_data)):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            total_reward += reward

            predictions.append(action)
            true_labels.append(prepared_test_data.iloc[i]['is_fraud'])

            if done:
                break

        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            'total_reward': total_reward,
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'f1_score': f1_score(true_labels, predictions),
        }

        # Add AUC if we have probability scores (not available for basic RL)
        # metrics['auc'] = roc_auc_score(true_labels, prediction_probs)

        return metrics

    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(path)

    def load_model(self, path: str):
        """Load a trained model"""
        if self.model_type == "PPO":
            self.model = PPO.load(path)
        elif self.model_type == "DQN":
            self.model = DQN.load(path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")