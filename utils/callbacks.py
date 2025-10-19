import logging
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

class RewardMetricsCallback(TrainerCallback):
    """Custom callback to log reward function metrics during training."""
    
    def __init__(self, reward_functions=None):
        self.reward_functions = reward_functions or []
        self.step_count = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log custom metrics after each training step."""
        self.step_count += 1
        
        # Log learning rate
        if hasattr(state, 'log_history') and state.log_history:
            last_log = state.log_history[-1]
            if 'learning_rate' in last_log:
                logger.info(f"Step {self.step_count}: LR = {last_log['learning_rate']:.2e}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Enhance logging with additional metrics."""
        if logs is None:
            return
            
        # Add custom metrics to the logs
        if 'train_loss' in logs:
            logs['custom/train_loss_smoothed'] = logs['train_loss']
        
        # Log reward function performance if available
        for i, reward_func in enumerate(self.reward_functions):
            if hasattr(reward_func, '__name__'):
                reward_name = reward_func.__name__
                # This would be populated if we had access to recent completions
                # For now, we log placeholder metrics that can be updated later
                logs[f'rewards/{reward_name}_avg'] = 0.5  # Placeholder