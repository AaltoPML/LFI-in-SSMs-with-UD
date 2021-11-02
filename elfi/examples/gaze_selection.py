'''Implementation of the gaze-based selection simulator
The original code By Xiuli Chen was adapted from here:
https://github.com/XiuliChen/GazeBasedSelection '''

from functools import partial

import numpy as np
import scipy.stats as ss
import scipy.io

import math
import gym
from gym import spaces
from stable_baselines3 import PPO

import elfi


def _calc_dis(p, q):
    ''' 
    calculate the Euclidean distance between points p and q 
    '''
    return np.sqrt(np.sum((p - q)**2))

def get_new_target(D):
    '''
    generate a target at a random angle, distance D away.
    '''
    angle = np.random.uniform(0, math.pi*2) 
    x_target = math.cos(angle) * D
    y_target = math.sin(angle) * D
    return np.array([x_target, y_target])


class GazeSelection(elfi.DynamicProcess):
    """Class for gaze-based selection simulator in ELFI. The simulator takes two 
    parameters that correspond to two type of noise that occur in the human eye
    behavior, trains a RL agent, and outputs the number of saccades (actions) it
    took to find the target on a display.

    Attributes
    ----------
    name : str
    target_name : str
    bounds : array
    true_params : list
        true_params[0] : ocular_noise, this parameter changes 
            according to log(t + 1)/10 + 0.1
        true_params[1] : spatial_noise
    _step : int
    """

    # epochs = 2e6
    def __init__(self, epochs=2e6, bounds=None, **kwargs):
        self.time_steps = epochs
        self.bounds = bounds or np.array([[30, 60], [0, 0.2], [0, 0.2]])
        self.target_name = 'log_d'
        self.name = 'Gaze selection' 

        super(GazeSelection, self).__init__(func=self.func, parameter_names = ['eye_latency', 'ocular_noise', 'spatial_noise'], **kwargs)
        self._step = 0
        
        self.belief_history = [None]



    def func(self, *params, n_obs=1, batch_size=1, random_state=None):
        """
        Stochastic function that performs simulations, given parameter values
        
        Parameters
        ----------
        params : array, 
            a batch of parameters, shape (batch_size, number_of_parameters)
        n_obs : int (optional), 
            number of observed data, equals to 1 in the experiments
        batch_size : int (optional), 
            number of function runs and results to return
        random_state : int (optional), 
            random seed for the stochastic function

        Returns
        -------
        results : array, 
            contains results of simulation (synthetic datasets) for each 
            parameter set, shape (batch_size, result_dimensions = 2)
        """        
        # initialize target width (w) and distance (d) in a series of experiments
        w_schuetz = np.array([1]) # 1.5, 2, 3, 4, 5])
        d_schuetz = np.array([5]) # , 10])
        unit = 0.5 / 10
        w_schuetz = np.round(w_schuetz * unit, 2)
        d_schuetz = np.round(d_schuetz * unit, 2)

        w = w_schuetz
        d = d_schuetz
        timesteps = self.time_steps

        results = list()
        sim_params = np.array( params ).reshape(self.param_dim, -1)
        batches = sim_params.shape[1]

        # print('Shape (func):', sim_params.shape)
        # print('Simulator call (gaze_selection.py : func):', sim_params)
        # for each parameter set in a batch
        for i in range(0, batches):
            # print('i batch: ', i)
            eye_latency = sim_params[0][i]
            ocular_std = sim_params[1][i]
            swapping_std = sim_params[2][i]

            for fitts_W in w:
                for fitts_D in d: 

                    params = np.array((fitts_D, fitts_W, ocular_std, swapping_std, timesteps))

                    # Instantiate the env
                    env = Gaze(fitts_W = fitts_W, 
                        fitts_D = fitts_D, 
                        ocular_std = ocular_std, 
                        swapping_std = swapping_std)

                    # Train the agent
                    model = PPO('MlpPolicy', env, verbose = 0, clip_range = 0.15)
                    model.learn(total_timesteps = int(timesteps))

                    # Test the trained agent
                    n_eps = n_obs
                    number_of_saccades = np.ndarray(shape = (n_eps, 1), dtype = np.float32)
                    movement_time_all = np.ndarray(shape = (n_eps, 1), dtype = np.float32)
                    eps = 0
                    while eps < n_eps:                
                        done = False
                        step = 0
                        obs = env.reset()
                        fixate = np.array([0, 0])
                        movement_time = 0
                        while not done:
                            step += 1
                            action, _ = model.predict(obs, deterministic = True)
                            obs, reward, done, info = env.step(action)
                            move_dis = _calc_dis(info['fixate'], fixate)
                            fixate = info['fixate']
                            movement_time += 2.7 * move_dis + eye_latency
                            if done:
                                number_of_saccades[eps] = step
                                movement_time_all[eps] = movement_time
                                eps += 1
                                break

            batch_result = [np.mean(movement_time_all)]
            results.append(batch_result)
        return np.array(results)


    def discrepancy(self, s, obs):
        """
        Weighted euclidean distance for each result dimension with coefficients 
        1 and 0.2;

        Parameters
        ----------
        s : array, 
            synthetic datasets
        obs : array, 
            observed dataset
        Returns
        -------
        dis : array, 
            discrepancy between the observed dataset and all synthetic datasets
        """        
        mov_obs = obs[0][0]
        dis = list()

        for entry in s:
            mov = entry[0]
            mov_dis = scipy.spatial.distance.euclidean(mov, mov_obs)
            dis.append(mov_dis)

        return np.array(dis)



    def create_model(self, observed):
        """
        Create model with new observed data and prior bounds
        Parameters
        ----------
        observed : array

        Returns
        -------
        model : elfi.ElfiModel
        """
        
        model = elfi.DynamicElfiModel()
        priors = [elfi.Prior(ss.uniform, self.bounds[i][0], 
            self.bounds[i][1] - self.bounds[i][0], model=model, name=self.param_names[i]) 
            for i in range(self.param_dim)]
        
        elfi.Simulator(self.sim_fn, *priors, observed=observed, name='Sim')

        if self.summarize:
            elfi.Summary(partial(np.mean, axis=1), model['Sim'], name='Mean')
            elfi.Summary(partial(np.std, axis=1), model['Sim'], name='Std')
            elfi.Distance('euclidean', model['Mean'], model['Std'], name='dist')
            elfi.Operation(np.log, model['dist'], name=self.target_name)
        else:
            elfi.Distance(self.discrepancy, model['Sim'], name='dist')
            elfi.Operation(np.log, model['dist'], name=self.target_name)

        return model


    def update_dynamic(self):
        """
        Update the true value of the dynamic component for the model.
        """
        self._step = self._step + 1
        self.true_params[0] = 12 * np.log(self._step) + 37
        self.true_params[1] = 0.01 #+ np.random.normal(0,0.001,1)[0]
        self.true_params[2] = 0.09 #+ np.random.normal(0,0.001,1)[0]
        # print('Update dynamics (true_parameters): ', self.true_params)

        








# This class describes the environment, which should remain unchanged
class Gaze(gym.Env):
    '''
    Description:
            The agent moves the eye to the target on the dispaly. 
            The agent's vision has spactial swapping noise that is a funtion 
            of the eccentricity

    States: the target position (type: Box(2, ));
            [-1,-1] top-left; [1,1] bottom-right 

    Actions: the fixation position (type: Box(2,));
            [-1,-1] top-left; [1,1] bottom-right 

    Observation: the estimate of where the target is based on one obs
            (type: Box(2, ));
            [-1,-1] top-left; [1,1] bottom-right 

    Belief: the estimate of where the target is based on all obs
            (type: Box(2, ));
            [-1,-1] top-left; [1,1] bottom-right 


    Reward:
            Reward of 0 is awarded if the eye reach the target.
            reward of -1 is awared if not


    Episode Termination:
            the eye reaches the target (within self.fitts_W/2)
            or reach the maximum steps

    '''

    def __init__(self, fitts_W, fitts_D, ocular_std, swapping_std):
        # task setting
        self.fitts_W = fitts_W
        self.fitts_D = fitts_D

        # agent ocular motor noise and visual spatial noise
        self.ocular_std = ocular_std
        self.swapping_std = swapping_std
         
        ## state, action and observation space
        # where to fixate and the target location
        self.action_space = spaces.Box(low = -1, high = 1, shape = (2, ), dtype = np.float64)
        self.state_space = spaces.Box(low = -1, high = 1, shape = (2, ), dtype = np.float64)

        #observation
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (2, ), dtype = np.float64)
        self.belief_space = spaces.Box(low = -1, high = 1, shape = (2, ), dtype = np.float64)

        # 50, 500
        self.max_fixation = 50

         
    def reset(self):
        # choose a random one target location
        self.state = get_new_target(self.fitts_D)

        # As in the experiments the participants starting with a 
        # fixation at the starting point, the agent start with the
        # first fixation at the center
        self.fixate = np.array([0, 0])
        self.n_fixation = 1

        # the first obs
        self.obs, self.obs_uncertainty = self._get_obs()

        self.belief, self.belief_uncertainty = self.obs, self.obs_uncertainty
        return self.belief


    def step(self, action):
        # execute the chosen action given the ocular motor noise
        move_dis = _calc_dis(self.fixate, action)
        ocular_noise = np.random.normal(0, self.ocular_std * move_dis, action.shape)
        self.fixate = action + ocular_noise
        self.fixate = np.clip(self.fixate, -1, 1)

        others={'n_fixation': self.n_fixation,
                'target': self.state, 
                'belief': self.belief,
                'aim': action,
                'fixate': self.fixate}

        self.n_fixation += 1

        # check if the eye is within the target region
        dis_to_target = _calc_dis(self.state, self.fixate)
        
        if dis_to_target < self.fitts_W / 2:
            done = True
            reward = 0

        # has not reached the target, get new obs at the new fixation location
        else:
            done = False
            reward = -1 
            self.obs, self.obs_uncertainty = self._get_obs()
            self.belief, self.belief_uncertainty = self._get_belief()

        if self.n_fixation > self.max_fixation:
            done = True

        return self.belief, reward, done, others


    def _get_obs(self):
        eccentricity = _calc_dis(self.state, self.fixate)
        obs_uncertainty = eccentricity
        spatial_noise = np.random.normal(0, self.swapping_std * eccentricity,
                                         self.state.shape)
        obs = self.state + spatial_noise
        # the obs should rarely be outside of -1 and 1, just in case
        '''
        if obs[0]>1 or obs[0]<-1 or obs[1]>1 or obs[0]<-1:
            print(obs)
            print('obs is out of the range!!!!!')
        '''
        obs = np.clip(obs, -1, 1)
        return obs, obs_uncertainty


    def _get_belief(self):
        z1, sigma1 = self.obs, self.obs_uncertainty
        z2, sigma2 = self.belief, self.belief_uncertainty

        w1 = sigma2**2 / (sigma1**2 + sigma2**2)
        w2 = sigma1**2 / (sigma1**2 + sigma2**2)

        belief = w1*z1 + w2*z2
        belief_uncertainty = np.sqrt( (sigma1**2 * sigma2**2)/(sigma1**2 + sigma2**2))

        return belief, belief_uncertainty
    

    def render(self):
        pass
