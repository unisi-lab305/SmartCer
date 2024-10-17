class RewardScaler:
    """
    Class for scaling and normalizing reward values in the CER environment.

    Attributes:
        mean (float): Running mean of the rewards.
        M2 (float): Accumulator for variance calculations.
        count (int): Count of rewards seen so far.
        epsilon (float): Small value to avoid division by zero in variance calculations.
        clip_min (float): Minimum value for clipping the scaled reward.
        clip_max (float): Maximum value for clipping the scaled reward.
        reset_interval (int, optional): Number of steps after which to reset statistics (None to disable resetting).
        step_counter (int): Counter for the number of updates since the last reset.
    """

    def __init__(self, clip_min: float = None, clip_max: float = None, reset_interval: int = None):
        """
        Initializes the RewardScaler.

        Args:
            clip_min (float, optional): Minimum value for clipping the scaled reward. Default is None.
            clip_max (float, optional): Maximum value for clipping the scaled reward. Default is None.
            reset_interval (int, optional): Number of steps after which to reset statistics (None to disable resetting).
        """
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0
        self.epsilon = 1e-8
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.reset_interval = reset_interval
        self.step_counter = 0

    def __repr__(self):
        return f'{self.__class__.__name__} → mean: {self.mean}, variance: {self.get_variance()}, count: {self.count}'

    def update(self, reward: float):
        """
        Updates the running mean and variance with a new reward.

        Args:
            reward (float): The new reward value to incorporate into the statistics.
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count     # update running mean
        delta2 = reward - self.mean
        self.M2 += delta * delta2           # update accumulator for variance
        self.step_counter += 1

        # check if it's time to reset the statistics
        if self.reset_interval and self.step_counter >= self.reset_interval:
            self.reset()

    def get_variance(self):
        """
        Returns the current variance of the rewards.

        Returns:
            float: Variance of the rewards; returns float('inf') if count < 2.
        """
        if self.count < 2:
            return float('inf')  # Variance is undefined for count < 2
        return self.M2 / self.count

    def scale(self, reward: float):
        """
        Scales and clips the given reward.

        Args:
            reward (float): The reward to be scaled.

        Returns:
            float: The scaled and clipped reward.
        """
        std = (self.get_variance()) ** 0.5   # Calculate standard deviation
        scaled_reward = (reward - self.mean) / (std + self.epsilon)  # Standardize reward
        # apply clipping to the scaled reward
        if self.clip_min is not None and self.clip_max is not None:
            scaled_reward = max(self.clip_min, min(scaled_reward, self.clip_max))
        return scaled_reward

    def reset(self):
        """
        Resets the statistics (mean, variance) to initial values.
        """
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0
        self.step_counter = 0
        print("Resetting reward statistics.")
