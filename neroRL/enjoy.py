"""
Instantiates an environment and loads a trained model based on the provided config.
The agent environment interaction is then shown in realtime for one episode on a specified seed.
Additionally a video can be rendered. Alternatively a website visualizing more properties, such as the value function,
can be generated.
"""

import logging
import torch
import numpy as np
import sys

from docopt import docopt
from gym import spaces

from neroRL.utils.yaml_parser import YamlParser
from neroRL.environments.wrapper import wrap_environment
from neroRL.utils.video_recorder import VideoRecorder
from neroRL.nn.actor_critic import create_actor_critic_model
from neroRL.normalization.observation_normalizer import NdNormalizer

# Setup logger
logging.basicConfig(level = logging.INFO, handlers=[])
logger = logging.getLogger("enjoy")
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(console)

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        nenjoy [options]
        nenjoy --help

    Options:
        --config=<path>            Path to the config file [default: ./configs/default.yaml].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --seed=<n>                 The to be played seed of an episode [default: 0].
        --num-episodes=<n>         The number of to be played episodes [default: 1].
        --video=<path>             Specify a path for saving a video, if video recording is desired. The file's extension will be set automatically. [default: ./video].
        --framerate=<n>            Specifies the frame rate of the to be rendered video. [default: 6]
        --generate_website         Specifies wether a website shall be generated. [default: False]
    """
    options = docopt(_USAGE)
    untrained = options["--untrained"]
    config_path = options["--config"]
    worker_id = int(options["--worker-id"])
    seed = int(options["--seed"])
    num_episodes = int(options["--num-episodes"])
    video_path = options["--video"]
    frame_rate = options["--framerate"]
    generate_website = options["--generate_website"]

    # Determine whether to record a video. A video is only recorded if the video flag is used.
    record_video = False
    for i, arg in enumerate(sys.argv):
        if "--video" in arg:
            record_video = True
            logger.info("Step 0: Video recording enabled. Video will be saved to " + video_path)
            logger.info("Step 0: Only 1 episode will be played")
            num_episodes = 1
            break

    if generate_website:
        logger.info("Step 0: Only 1 episode will be played")
        num_episodes = 1

    # Load environment, model, evaluation and training parameters
    configs = YamlParser(config_path).get_config()

    # Determine cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # Launch environment
    logger.info("Step 1: Launching environment")
    configs["environment"]["reset_params"]["start-seed"] = seed
    configs["environment"]["reset_params"]["num-seeds"] = 1
    configs["environment"]["reset_params"]["seed"] = seed
    env = wrap_environment(configs["environment"], worker_id, realtime_mode = True, no_graphics=False, record_trajectory = record_video or generate_website)
    # Retrieve observation space
    visual_observation_space = env.visual_observation_space
    vector_observation_space = env.vector_observation_space
    #changed here 
    if isinstance(env.action_space, spaces.Box):
        action_space_shape = env.action_space.shape

    # Build or load model
    logger.info("Step 2: Creating model")
    share_parameters = False
    if configs["trainer"]["algorithm"] == "PPO":
        share_parameters = configs["trainer"]["share_parameters"]
    model = create_actor_critic_model(configs["model"], share_parameters, visual_observation_space,
                            vector_observation_space, action_space_shape,
                            configs["model"]["recurrence"] if "recurrence" in configs["model"] else None, device)

    observationNormalizer = NdNormalizer(vector_observation_space) if configs["model"]["normalize_observations"] else None

    if not untrained:
        logger.info("Step 2: Loading model from " + configs["model"]["model_path"])
        checkpoint = torch.load(configs["model"]["model_path"], map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "recurrence" in configs["model"]:
            model.set_mean_recurrent_cell_states(checkpoint["hxs"], checkpoint["cxs"])
        if "normalizer" in checkpoint.keys():
            observationNormalizer.set_data(checkpoint["normalizer"])

    model.eval()

    # Run all desired episodes
    # Reset environment
    logger.info("Step 3: Resetting the environment")
    logger.info("Step 3: Using seed " + str(seed))
    vis_obs, vec_obs, agent_ids, actions_next_step = env.reset(configs["environment"]["reset_params"])
    n_agents = env.n_agents
    for i, vec_ob in enumerate(vec_obs):
        vec_obs[i] = observationNormalizer.forward(vec_ob) if observationNormalizer is not None else vec_ob
    agent_id_map = env.agent_id_map
    done = 0

    # Play episode
    logger.info("Step 4: Run " + str(num_episodes) + " episode(s) in realtime . . .")

    # Play one episode
    with torch.no_grad():
        while not done >= num_episodes:
            # Forward the neural net
            vec_obs = torch.tensor(vec_obs, dtype=torch.float32, device=device) if vec_obs is not None else None
            policy, value, recurrent_cell, _ = model(vis_obs, vec_obs, recurrent_cell=None)

            action = policy.sample()
  
            action_np = action.cpu().numpy()
  
            pass_actions = np.zeros((actions_next_step,)+(action_space_shape))
            for i,a in enumerate(agent_ids[:actions_next_step]):
                pass_actions[i] = action_np[agent_id_map[a]]

            # Step environment
            vis_obs, temp_vec_obs, rewards, agent_ids, actions_next_step, episode_end_info = env.step(pass_actions)

            vec_obs = np.zeros_like(vec_obs)
            for x in reversed(range(0, len(agent_ids))):
                agent_id = agent_ids[x]
                agent_index = agent_id_map[agent_id]
                if temp_vec_obs[x].shape is (): # after the initial reset, the get_steps method of the environments return terminal steps with empty observations (shape () ). These are ignored using this if statement
                    continue

                vec_obs[agent_index] = observationNormalizer.forward(temp_vec_obs[x]) if observationNormalizer is not None else temp_vec_obs[x]

                if x >= actions_next_step:
                    #dones[agent_index] = True
                    done += 1
                    logger.warning("Finished Episode " + str(done) + " by agent with id " + str(agent_id))
                    logger.info("-> Episode Reward " + str(episode_end_info[x-actions_next_step]["reward"]))
                    logger.info("-> Episode Length " + str(episode_end_info[x-actions_next_step]["length"]))
                

    # logger.info("Episode Reward: " + str(info["reward"]))
    # logger.info("Episode Length: " + str(info["length"]))


    env.close()

if __name__ == "__main__":
    main()
    