import random
from animalai.envs.environment import AnimalAIEnvironment
from animalai.envs.arena_config import ArenaConfig 
from gym_unity.envs import UnityToGymWrapper

def load_env(config_file, args, training = True):
    return AnimalAIEnvironment(
        file_name = 'env/AnimalAI',
        base_port = 5005 + random.randint(0, 100),
        arenas_configurations = ArenaConfig(config_file),
        play = False,
        useCamera = False,
        useRayCasts = True,
        raysPerSide = 1,
        rayMaxDegrees = 90,
        decisionPeriod = 3 * args.frame_skips,
        seed = args.seed,
        inference = False,
        targetFrameRate = -1 if training else 60,
        captureFrameRate = 0 if training else 60
    )

def get_env(config_file, args, training):
    env = load_env(config_file, args, training)
    return UnityToGymWrapper(env, uint8_visual=False, allow_multiple_obs=False, flatten_branched=True)
