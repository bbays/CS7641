import numpy as np
from hiive.visualization import mdpviz


class FireManagementSpec:

    def __init__(self, population_classes=7, fire_classes=13, seed=200972, verbose=True):
        self.seed = seed
        self.verbose = verbose
        self.population_classes = population_classes
        self.fire_classes = fire_classes
        self.states = {}

        self.spec = mdpviz.MDPSpec()

        self._action_do_nothing = self.spec.action('do_nothing')
        self._action_burn = self.spec.action('burn')

        self._probabilities = {}
        self.name = f'fire_management_{population_classes}_{fire_classes}_{seed}'
        self.n_actions = 2
        self.n_states = self.fire_classes * self.population_classes

        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self._setup_mdp()

    def _reset_state_probabilities(self):
        self._probabilities = {}

    def _get_probability_for_state(self, pc, fc):
        state_name = self._get_state_name(pc, fc)
        if state_name not in self._probabilities:
            return None
        return self._probabilities[state_name]

    def _set_probability_for_state(self, pc, fc, p):
        state_name = self._get_state_name(pc, fc)
        if state_name not in self._probabilities:
            self._probabilities[state_name] = 0.
        self._probabilities[state_name] += p
        return self._probabilities[state_name]

    @staticmethod
    def _is_terminal(s):
        return False  # s == 0

    @staticmethod
    def get_habitat_suitability(years):
        if years < 0:
            msg = "Invalid years '%s', it should be positive." % str(years)
            raise ValueError(msg)
        if years <= 5:
            return 0.2 * years
        elif 5 <= years <= 10:
            return -0.1 * years + 1.5
        else:
            return 0.5

    @staticmethod
    def _get_state_name(pc, fc):
        return f'pc:{pc}, fc:{fc}'

    def _get_state(self, pc, fc):
        state_name = self._get_state_name(pc, fc)
        is_terminal = self._is_terminal(pc)
        if state_name not in self.states:
            state = self.spec.state(name=state_name, terminal_state=is_terminal)
            self.states[state_name] = state
        # print(f'{state_name} : {is_terminal}')
        state = self.states[state_name]
        return state

    def _add_state_transition_and_reward(self, pc, fc, action):
        cs = self._get_state(pc, fc)
        results = self._get_reward_and_new_state_values(pc, fc, action)
        for reward, npc, nfc, tp in results:
            ns = self._get_state(npc, nfc)
            ns = mdpviz.NextState(state=ns, weight=tp)
            self.spec.transition(state=cs, action=action, outcome=ns)
            self.spec.transition(state=cs, action=action, outcome=mdpviz.Reward(reward))
            if self.verbose:
                print(f'[state:action]: [{(pc, fc)} : {action.name}] -> new state: {(npc, nfc)}, '
                      f'p(t): {tp}, reward: {reward} ')

    def transition_fire_class(self, fc, action):
        if action == self._action_do_nothing:
            return (fc + 1) if fc < self.fire_classes - 1 else fc
        elif action == self._action_burn:
            return 0
        return fc

    def _get_reward_and_new_state_values(self, pc, fc, action, default_p=0.5):
        pop_change_down = -1
        pop_change_same = 0

        self._probabilities = {}
        transition_probability_up = None
#        if pc == 1 and fc == 0 and action == self._action_burn:
#            print()

        r = self.get_habitat_suitability(fc)
        fc = self.transition_fire_class(fc, action)
        if pc == 0:
            # dead
            return [[0.0, 0, fc, 1.0]]  # stays in same state
        if pc == self.population_classes - 1:
            pop_change_up = 0
            if action == self._action_burn:
                pop_change_same -= 1
                pop_change_down -= 1

            tsd = self._set_probability_for_state(pc=pc + pop_change_down,
                                                  fc=fc,
                                                  p=(1.0 - default_p) * (1.0 - r))
            tss = self._set_probability_for_state(pc=pc + pop_change_same,
                                                  fc=fc,
                                                  p=1.0 - tsd)
        else:
            # Population abundance class can stay the same, transition up, or
            # transition down.
            pop_change_same = 0
            pop_change_up = 1
            pop_change_down = -1

            # If action 1 is taken, then the patch is burned so the population
            # abundance moves down a class.
            if action == self._action_burn:
                pop_change_same -= 1
                pop_change_up -= 1
                pop_change_down -= (1 if pop_change_down > 0 else 0)

            tss = self._set_probability_for_state(pc=pc + pop_change_same,
                                                  fc=fc,
                                                  p=default_p)

            tsu = self._set_probability_for_state(pc=pc + pop_change_up,
                                                  fc=fc,
                                                  p=(1 - default_p)*r)
            # In the case when transition_down = 0 before the effect of an action
            # is applied, then the final state is going to be the same as that for
            # transition_same, so we need to add the probabilities together.
            tsd = self._set_probability_for_state(pc=pc + pop_change_down,
                                                  fc=fc,
                                                  p=(1 - default_p)*(1 - r))

        # build results
        results = []

        npc_up = pc + pop_change_up
        npc_down = pc + pop_change_down
        npc_same = pc + pop_change_same

        transition_probabilities = {
            (npc_up, self._get_probability_for_state(npc_up, fc)),
            (npc_down, self._get_probability_for_state(npc_down, fc)),
            (npc_same, self._get_probability_for_state(npc_same, fc))
        }

        for npc, probability in transition_probabilities:
            if probability is not None and probability > 0.0:
                reward = int(npc > 0)
                results.append((reward, npc, fc, probability))

        return results

    # noinspection PyStatementEffect
    def _setup_mdp(self):
        # build transitions
        for pc in range(0, self.population_classes):
            if self._is_terminal(pc):
                continue
            for fc in range(0, self.fire_classes):
                # actions
                self._add_state_transition_and_reward(pc=pc, fc=fc, action=self._action_do_nothing)
                self._add_state_transition_and_reward(pc=pc, fc=fc, action=self._action_burn)
                if self.verbose:
                    print()

    def get_transition_and_reward_arrays(self, p_default=0.5):
        return self.spec.get_transition_and_reward_arrays(p_default)

    def to_graph(self):
        return self.spec.to_graph()

    def to_env(self):
        return self.spec.to_discrete_env()