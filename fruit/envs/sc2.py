from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env


import random
import numpy as np
from fruit.envs.base import BaseEnvironment


_NO_OP              = 0
_HARVEST_GATHER     = 1
_BUILD_SUPPLY_DEPOT = 2
_BUILD_BARRACKS     = 3
_TRAIN_MARINE       = 4
_ATTACK_MINIMAP     = 5

# Please note that this environment is very basic, users should add more features for it
class SC2Environment(BaseEnvironment):
    def __init__(self,map_name='Simple64',
                 resolution=64,
                 step_mul = 48,
                 render=False,
                 max_frames = 0,
                 id = 1, ):
        self.map_name = map_name
        self.resolution = resolution
        self.step_mul = step_mul

        self.timesteps = None
        self.totalframes = 0
        self.max_frames = max_frames
        self.terminal_state = False

        self.num_of_objs = 1
        self.id = id
        self.initialize()

    def get_game_name(self):
        return 'SC2 ENVIRONMENT'

    def initialize(self):
        self.agents = SmartAgent()
        self.env = sc2_env.SC2Env(
            map_name=self.map_name,
            players=[
                     sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot(sc2_env.Race.terran,
                                 sc2_env.Difficulty.very_easy),
                     ],
            agent_interface_format=features.AgentInterfaceFormat(
                action_space  = actions.ActionSpace.RAW,
                use_raw_units = True,
                raw_resolution= self.resolution,
            ),
            step_mul= self.step_mul,
            disable_fog=True,
        )

        observation_spec = self.env.observation_spec()
        action_spec = self.env.action_spec()
        # for agent, obs_spec, act_spec in zip(self.agents, observation_spec, action_spec):
        self.agents.setup(observation_spec, action_spec)

        self.terminal_state = False
        self.timesteps = self.env.reset()
        self.agents.reset()


    def clone(self):
        return SC2Environment(
                 # agents         =  agents,
                 map_name       =  self.map_name,
                 action_space   =  self.action_space,
                 resolution     =  self.resolution,
                 step_mul       =  self.step_mul,
                 max_frames     =  self.max_frames,
                 id = self.id + 1,
        )

    def get_number_of_objectives(self):
        return self.num_of_objs

    def get_seed(self):
        pass

    def reset(self):
        self.terminal_state = False
        self.timesteps = self.env.reset()
        self.agents.reset()


    def step(self, actions):
        self.totalframes += 1
        if self.max_frames and self.total_frames >= self.max_frames:
            self.timesteps = self.env.reset()
            self.agents.reset()

        if self.timesteps[0].last():
            self.timesteps = self.env.reset()
            self.agents.reset()

        actions = [self.agents.step(actions,self.timesteps[0])]
        self.timesteps = self.env.step(actions)
        return self.timesteps[0].reward

    def get_current_steps(self):
        return self.timesteps[0]

    def step_all(self, action):
        reward = self.step(action)
        current_state = self.get_state()
        terminal = self.is_terminal()
        # return self.current_state, reward, terminal, None
        return current_state, reward, terminal, None

    def get_state_space(self):
        from fruit.types.priv import Space
        shape = (20, 1)
        min_value = np.zeros(shape)
        max_value = np.full(shape, 100)
        return Space(min_value, max_value, True)

    def get_action_space(self):
        from fruit.types.priv import Space
        return Space(0, 5, True)

    def get_state(self):
        return self.agents.get_state(self.timesteps[0])

    def is_terminal(self):
        return self.terminal_state

    def is_render(self):
        return self.should_render

    def get_number_of_agents(self):
        return len(self.agents)


class Agent(base_agent.BaseAgent):
    actions = ("do_nothing",
               "harvest_minerals",
               "build_supply_depot",
               "build_barracks",
               "train_marine",
               "attack")

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def step(self, obs):
        super(Agent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]
            scv = random.choice(idle_scvs)
            distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and
                len(scvs) > 0):
            supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
            distances = self.get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and len(barrackses) == 0 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):
            barracks_xy = (22, 21) if self.base_top_left else (35, 45)
            distances = self.get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0):
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            distances = self.get_distances(obs, marines, attack_xy)
            marine = marines[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        actions.RAW_FUNCTIONS.no_op()
        return actions.RAW_FUNCTIONS.no_op()


class SmartAgent(Agent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.new_game()

    def reset(self):
        super(SmartAgent, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length
                          if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)


        return (len(command_centers),
                len(scvs),
                len(idle_scvs),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                queued_marines,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine,
                len(enemy_command_centers),
                len(enemy_scvs),
                len(enemy_idle_scvs),
                len(enemy_supply_depots),
                len(enemy_completed_supply_depots),
                len(enemy_barrackses),
                len(enemy_completed_barrackses),
                len(enemy_marines))

    def step(self, action, obs):
        super(SmartAgent, self).step(obs)

        if int(action.__str__()) == _NO_OP:
            return self.do_nothing(obs=obs)
        elif int(action.__str__()) == _HARVEST_GATHER:
            return self.harvest_minerals(obs=obs)
        elif int(action.__str__()) == _BUILD_SUPPLY_DEPOT:
            return self.build_supply_depot(obs=obs)
        elif int(action.__str__()) == _BUILD_BARRACKS:
            return self.build_barracks(obs=obs)
        elif int(action.__str__()) == _TRAIN_MARINE:
            return self.train_marine(obs=obs)
        elif int(action.__str__()) == _ATTACK_MINIMAP:
            return self.attack(obs=obs)


def get_random_action(is_discrete, action_range, action_space):
    if is_discrete:
        if len(action_range) == 2 and isinstance(action_range[0], (list, np.ndarray, tuple)):
            action = [random.randint(action_range[0][i], action_range[1][i]) for i in range(len(action_range[0]))]
        else:
            action = random.randint(0, len(action_range) - 1)
    else:
        rand = np.random.rand(*tuple(action_space.get_shape()))[0]
        action = np.multiply(action_range[1] - action_range[0], rand) + action_range[0]
    return action

def train_random_agent():
    environment = SC2Environment()
    state_space = environment.get_state_space()
    action_space = environment.get_action_space()

    is_discrete = False
    action_range = None
    if isinstance(action_space, tuple):
        for s in action_space:
            action_range, is_discrete = s.get_range()
            print(action_range, s.get_shape())
    else:
        action_range, is_discrete = action_space.get_range()
        print(action_range, action_space.get_shape())

    environment.reset()
    for i in range(1000):
        if isinstance(action_space, tuple):
            action = []
            for s in action_space:
                action_range, is_discrete = s.get_range()
                action.append(get_random_action(is_discrete, action_range, s))
        else:
            action = get_random_action(is_discrete, action_range, action_space)

        reward = environment.step(action)
        next_state = environment.get_state()
        state = next_state
        terminal = environment.is_terminal()
        print(action,reward)

        if terminal:
            environment.reset()
            break

if __name__ == '__main__':
    import sys
    from absl import flags

    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    train_random_agent()
    print("Done")

