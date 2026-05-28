import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from navstack.balancing.gym_env import SegwayEnv
import time

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LOG_DIR = "logs"

def train(timesteps=100000):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    print("Initializing Environment...")
    env = SegwayEnv()
    check_env(env) # Validate Gym Interface

    print("Setting up PPO Agent...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    
    print(f"Training for {timesteps} steps...")
    model.learn(total_timesteps=timesteps)
    
    save_path = f"{MODELS_DIR}/ppo_segway"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

def test():
    print("Loading Model...")
    model_path = f"{MODELS_DIR}/ppo_segway.zip"
    if not os.path.exists(model_path):
        print("Model not found! Train first.")
        return
        
    model = PPO.load(model_path)
    env = SegwayEnv()
    
    obs, info = env.reset()
    done = False
    
    print("Running Inference Loop...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render (Not implemented in Gym due to complexity, relying on internal plots)
        # For now, we print state
        print(f"Tilt: {obs[2]:.4f} rad | Torque: {action[0]:.2f} Nm")
        
        # Slow down for visualization
        time.sleep(0.01)
        
        if terminated or truncated:
            obs, info = env.reset()
            print("Episode Reset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--test", action="store_true", help="Test the trained agent")
    parser.add_argument("--steps", type=int, default=100000, help="Training timesteps")
    args = parser.parse_args()
    
    if args.train:
        train(args.steps)
    elif args.test:
        test()
    else:
        print("Use --train or --test")
