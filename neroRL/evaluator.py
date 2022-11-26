import numpy as np
import torch
import time

from neroRL.utils.worker import Worker
from neroRL.utils.video_recorder import VideoRecorder
from neroRL.normalization.observation_normalizer import NdNormalizer
class Evaluator():
    """Evaluates a model based on the initially provided config."""
    def __init__(self, configs, worker_id, visual_observation_space, vector_observation_space, action_space_shape, n_agents, video_path = "video", record_video = False, frame_rate = 1, generate_website = False, observationNormalizer = None):
        """Initializes the evaluator and its environments
        
        Arguments:
            eval_config {dict} -- The config of the evaluation
            env_config {dict} -- The config of the environment
            worker_id {int} -- The offset of the port to communicate with the environment
            visual_observation_space {box} -- Visual observation space of the environment
            vector_observation_space {tuple} -- Vector observation space of the environment
        """
        # Set members
        self.configs = configs
        self.n_workers = configs["evaluation"]["n_workers"]
        self.n_agents = n_agents
        self.seeds = configs["evaluation"]["seeds"]
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space
        self.action_space_shape = action_space_shape
        self.video_path = video_path
        self.record_video = record_video
        self.frame_rate = frame_rate
        self.generate_website = generate_website

        self.observationNormalizer = observationNormalizer

        # Launch environments
        self.workers = []
        for i in range(self.n_workers):
            id = worker_id + i + 200 - self.n_workers
            self.workers.append(Worker(configs["environment"], id, record_video = record_video))

        # Check for recurrent policy
        self.recurrence = None if not "recurrence" in configs["model"] else configs["model"]["recurrence"]

    def evaluate(self, model, device):
        """Evaluates a provided model on the already initialized evaluation environments.

        Arguments:
            model {nn.Module} -- The to be evaluated model
            device {torch.device} -- The to be used device for executing the model

        Returns:
            eval_duration {float} -- The duration of the completed evaluation
            episode_infos {dict} -- The raw results of each evaluated episode
        """
        time_start = time.time()
        episode_infos = []
        # Loop over all seeds
        for seed in self.seeds:
            print("starting new seed")
            # Initialize observations
            vis_obs = None
            
            if self.vector_observation_space is not None:
                vec_obs = np.zeros((self.n_workers,self.n_agents) + self.vector_observation_space, dtype=np.float32)
            else:
                vec_obs = None
            
            self.agent_id_map = [{} for _ in self.workers]
            self.actions_next_step = [ [] for _ in self.workers]
            
            # Reset workers and set evaluation seed
            for worker in self.workers:
                reset_params = self.configs["environment"]["reset_params"]
                reset_params["start-seed"] = seed
                reset_params["num-seeds"] = 1
                worker.child.send(("hard_reset", reset_params))
            # Grab initial observations
            for w, worker in enumerate(self.workers):
                vis, vec, agent_ids, actions_next_step = worker.child.recv()
                if vis_obs is not None:
                    vis_obs[w] = vis
                if vec_obs is not None:
                    vec_obs[w] = vec
                for agent_id in agent_ids:
                    self.agent_id_map[w][agent_id] = len(self.agent_id_map[w].items())

                self.actions_next_step[w] = list(agent_ids[:actions_next_step])           

            # Every worker plays its episode
            dones = np.zeros((self.n_workers,self.n_agents))

            with torch.no_grad():
                while not np.all(dones):
                    #print("stepping")
                    # Sample action and send it to the worker if not done
                    for w, worker in enumerate(self.workers):
                        if not np.all(dones[w]):
                            # While sampling data for training we feed batches containing all workers,
                            # but as we evaluate entire episodes, we feed one worker at a time
                            vis_obs_batch = torch.tensor(vis_obs[w], dtype=torch.float32, device=device) if vis_obs is not None else None
                            vec_obs_batch = torch.tensor(vec_obs[w], dtype=torch.float32, device=device) if vec_obs is not None else None

                            policy, value,_, _ = model(vis_obs_batch, vec_obs_batch, None)

                            pass_actions = np.zeros((len(self.actions_next_step[w]),)+(self.action_space_shape))
                            # _probs = np.zeros((len(self.actions_next_step[x]),)+(self.action_space_shape))
                            #entropy = np.zeros((len(self.actions_next_step[x]),)+(self.action_space_shape))

                            # Sample action
                            action = policy.sample()

                            for i,a in enumerate(self.actions_next_step[w]):
                                pass_actions[i] = action[self.agent_id_map[w][a]]
                            #_probs.append(action.probs)
                            #entropy.append(action.entropy().item())


                            # Step environment
                            worker.child.send(("step", pass_actions))

                    # Receive and process step result if not done
                    for w, worker in enumerate(self.workers):
                        if not np.all(dones[w]):
                            vis_obs, temp_vec_obs, _, agent_ids, actions_next_step, episode_end_info = worker.child.recv()  
                            
                            self.actions_next_step[w] = list(agent_ids[:actions_next_step])

                            for x in reversed(range(0, len(agent_ids))):
                                agent_id = agent_ids[x]
                                agent_index = self.agent_id_map[w][agent_id]

                                if temp_vec_obs[x].shape is (): # after the initial reset, the get_steps method of the environments return terminal steps with empty observations (shape () ). These are ignored using this if statement
                                    continue

                                vec_obs[w][agent_index] = self.observationNormalizer.forward(temp_vec_obs[x]) if self.observationNormalizer is not None else temp_vec_obs[x]
                             
                                if x >= actions_next_step:
                                    if not dones[w, agent_index]:
                                        info = episode_end_info[x-actions_next_step]
                                        info["seed"] = seed
                                        episode_infos.append(info)                                   
                                        dones[w, agent_index] = 1    
        
                print("Finished Seed")
            # Seconds needed for a whole update
            time_end = time.time()
            eval_duration = int(time_end - time_start)

        # Return the duration of the evaluation and the raw episode results
        return eval_duration, episode_infos

    def close(self):
        """Closes the Evaluator and destroys all worker."""
        for worker in self.workers:
                worker.child.send(("close", None))
