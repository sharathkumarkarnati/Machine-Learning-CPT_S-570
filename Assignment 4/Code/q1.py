import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.setrecursionlimit(20000)


class Env:
    def __init__(self):
        self.grid_world = np.zeros((10, 10))
        self.wall = [
            [3, 2],
            [3, 3],
            [3, 4],
            [3, 5],
            [3, 7],
            [3, 8],
            [3, 9],
            [4, 5],
            [5, 5],
            [6, 5],
            [7, 5],
            [8, 5],
        ]
        self.reward = {
            (4, 4): -1,
            (5, 6): -1,
            (5, 7): -1,
            (6, 6): 1,
            (6, 7): -1,
            (6, 9): -1,
            (7, 9): -1,
            (8, 4): -1,
            (8, 6): -1,
            (8, 7): -1,
        }
        self.goal = [6, 6]


class QLearning:
    def __init__(self):
        self.current_location = None
        self.beta = 0.9
        self.actions = {"L": 1, "R": 2, "U": 3, "D": 4}
        self.actions_ = ["L", "R", "U", "D"]
        self.env = Env()
        self.epsilon = 0.1
        self.alpha = 0.01
        self.q = np.zeros((10, 10, 4))
        self.n_visits = np.zeros((10, 10))
        self.start_temp = 5

    def init_agent(self):
        self.current_location = [1, 1]
        self.total_reward = 0

    def get_action(self, exploration="e-greedy"):
        if exploration == "e-greedy":
            if self.epsilon > np.random.random():
                action = self.get_move_egreedy(greedy=False)
            else:
                action = self.get_move_egreedy()
        else:
            action = self.get_move_boltzmann()

        return action

    def get_new_state(self, action):
        new_state = self.current_location[:]

        if action == self.actions["L"]:
            new_state[1] -= 1
        elif action == self.actions["R"]:
            new_state[1] += 1
        elif action == self.actions["U"]:
            new_state[0] -= 1
        elif action == self.actions["D"]:
            new_state[0] += 1

        return new_state

    def act(self, exploration="e-greedy"):
        self.n_visits[self.current_location[0] - 1, self.current_location[1] - 1] += 1
        action = self.get_action(exploration)
        state = [self.current_location[0], self.current_location[1]]
        state_ = self.get_new_state(action + 1)
        if state_ != self.env.goal:
            self.current_location = state_
            self.act()
        if (state_[0], state_[1]) in self.env.reward.keys():
            reward = self.env.reward[(state_[0], state_[1])]
        else:
            reward = 0
        self.total_reward += reward
        max_q_next = np.max(self.q[state_[0] - 1, state_[1] - 1])

        self.q[state[0] - 1, state[1] - 1, action] += self.alpha * (
            reward + self.beta * max_q_next - self.q[state[0] - 1, state[1] - 1, action]
        )

    def get_move_egreedy(self, greedy=True):
        moves = self.valid_moves()
        if greedy:
            max_value = max(
                self.q[
                    self.current_location[0] - 1, self.current_location[1] - 1, moves
                ]
            )
            if max_value == 0:
                moves = [
                    moves[i]
                    for i in np.where(
                        self.q[
                            self.current_location[0] - 1,
                            self.current_location[1] - 1,
                            moves,
                        ]
                        == 0
                    )[0]
                ]
            return np.random.choice(moves)
        else:
            return moves[np.random.randint(0, len(moves))]

    def get_move_boltzmann(self):
        T = (
            self.start_temp
            / self.n_visits[self.current_location[0] - 1, self.current_location[1] - 1]
        )
        moves = self.valid_moves()
        q_values = self.q[
            self.current_location[0] - 1, self.current_location[1] - 1, moves
        ]
        prob = np.exp(q_values / T) / sum(np.exp(q_values / T))
        return moves[np.argmax(prob)]

    def valid_moves(self):
        moves = []
        if (
            self.current_location[1] > 1
            and [self.current_location[0], self.current_location[1] - 1]
            not in self.env.wall
        ):
            moves.append(self.actions["L"] - 1)

        if (
            self.current_location[1] < 10
            and [self.current_location[0], self.current_location[1] + 1]
            not in self.env.wall
        ):
            moves.append(self.actions["R"] - 1)

        if (
            self.current_location[0] > 1
            and [self.current_location[0] - 1, self.current_location[1]]
            not in self.env.wall
        ):
            moves.append(self.actions["U"] - 1)

        if (
            self.current_location[0] < 10
            and [self.current_location[0] + 1, self.current_location[1]]
            not in self.env.wall
        ):
            moves.append(self.actions["D"] - 1)

        return moves

    def get_current_policy(self):
        policy = np.zeros((10, 10))
        for i in range(self.q.shape[0]):
            for j in range(self.q.shape[1]):
                if [i + 1, j + 1] in self.env.wall:
                    policy[i, j] = 0
                else:
                    policy[i, j] = np.argmax(self.q[i, j, :])
        return policy

    def display_grid(self):
        for i in range(self.q.shape[0]):
            for j in range(self.q.shape[1]):
                if [i + 1, j + 1] in self.env.wall:
                    print("  w   |", end="")
                elif [i + 1, j + 1] == self.env.goal:
                    print(" g=1  |", end="")
                elif (i + 1, j + 1) in self.env.reward.keys():
                    print(str("%.3f" % self.env.reward[(i + 1, j + 1)]) + "|", end="")
                else:
                    max_val = max(self.q[i, j, :])
                    if max_val >= 0:
                        print(str(" %.3f" % round(max_val, 3)) + "|", end="")
                    else:
                        print(str("%.3f" % round(max_val, 3)) + "|", end="")
            print()

    def action_symbol(self, action):
        if action == self.actions["L"]:
            return "<"
        elif action == self.actions["R"]:
            return ">"
        elif action == self.actions["U"]:
            return "^"
        elif action == self.actions["D"]:
            return "v"

    def check_convergence(self, prev_policy):
        policy = self.get_current_policy()
        return np.sum((policy - prev_policy) ** 2)

    def display_policy(self):
        for i in range(self.q.shape[0]):
            for j in range(self.q.shape[1]):
                if [i + 1, j + 1] in self.env.wall:
                    print("  w  |", end="")
                elif [i + 1, j + 1] == self.env.goal:
                    print(" g=1 |", end="")
                elif (i + 1, j + 1) in self.env.reward.keys():
                    max_action = np.argmax(self.q[i, j, :])
                    print(
                        str(" %.0f" % self.env.reward[(i + 1, j + 1)])
                        + ","
                        + self.action_symbol(max_action + 1)
                        + "|",
                        end="",
                    )
                else:
                    max_action = np.argmax(self.q[i, j, :])
                    print("   " + self.action_symbol(max_action + 1) + " |", end="")
            print()

    def plot_figure(self, values, y_limit, title, y_label, x_label, save_as):
        print("Plotting graph")
        fig, ax = plt.subplots(figsize=(15, 12))
        ax.plot(values, "r")
        ax.set_ylim(y_limit[0], y_limit[1])
        ax.set_title(title, fontsize=24)
        ax.set_ylabel(y_label, fontsize=18)
        ax.set_xlabel(x_label, fontsize=18)
        print("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 4/Output/" + save_as + ".jpg")
        plt.show()
        fig.savefig("/Users/sharathkarnati/Desktop/ML/KARNATI-011852253/Assignment 4/Output/" + save_as + ".jpg")


if __name__ == "__main__":
    print("***** Qlearning for Gridworld *****\n")
    print("For each experiment it runs 20,000 iterations")
    print("Printing results at every 5000 iterations\n")
    experiments = [
        ["e-greedy", 0.1],
        ["e-greedy", 0.2],
        ["e-greedy", 0.3],
        ["boltzmann", 1],
        ["boltzmann", 5],
        ["boltzmann", 10],
        ["boltzmann", 20],
    ]
    for experiment in experiments:
        print("\n***** Experiment: " + str(experiment) + " *****\n")
        agent = QLearning()
        if experiment[0] == "e-greedy":
            agent.epsilon = experiment[1]
        else:
            agent.start_temp = experiment[1]

        rewardsPerEpisode = []
        policyDiff = []
        itr = 0
        while True:
            agent.init_agent()
            itr += 1

            agent.act(exploration=experiment[0])

            if itr > 1:
                policyDiff.append(agent.check_convergence(prev_policy))

            prev_policy = agent.get_current_policy()

            if itr % 5000 == 0:
                print("\n Iteration:" + str(itr))
                agent.display_grid()
                agent.display_policy()
            if itr == 20000:
                agent.plot_figure(
                    policyDiff,
                    [min(policyDiff) - 2, max(policyDiff) + 2],
                    "Change in Policy per Episode, Experiment:" + str(experiment),
                    "Change (squared)",
                    "Episode",
                    str(experiment[0]) + "_" + str(experiment[1]),
                )
                break
            rewardsPerEpisode.append(agent.total_reward)
