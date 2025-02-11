import gym
from stable_baselines3 import PPO

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Initialize the PPO (Proximal Policy Optimization) model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Test the trained model
episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()  # This will show the game visually
        action, _states = model.predict(obs)  # AI decides action
        obs, reward, done, info = env.step(action)  # Perform action
        score += reward

    print(f"Episode: {episode}, Score: {score}")

env.close()
