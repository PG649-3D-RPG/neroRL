from neroRL.environments.gym_continuous_wrapper import GymContinuousWrapper

env = GymContinuousWrapper("MountainCarContinuous-v0")

for _ in range(10):
    _, vec_obs = env.reset()
    done = False
    while not done:
        _, vec_obs, reward, done, info = env.step(env.action_space.sample())
    if info:
        print("Episode reward: " + str(info["reward"]))
        print("Episode length: " + str(info["length"]))