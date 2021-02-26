import numpy as np
import torch

from sumtree import SumTree


class ReplayBuffer:
    """
    Uniformly random sample transitions
    """
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transtions(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(rewards), \
               torch.from_numpy(states_), torch.from_numpy(dones)

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)


class PERMemory:
    epsilon = 0.0001
    alpha = 0.6
    size = 0

    # SumTreeについては参考にしたブログから拝借して必要なものをつけたし
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    # Proportional prioritizationによるpriorityの計算
    def _getPriority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    # 新しい経験を入れる際は、必ず一度はreplyされるようにその時点で最大のpriorityで
    # reply開始前の場合は論文に従いpriority=1とした
    def push(self, transition):
        self.size += 1

        priority = self.tree.max()
        if priority <= 0:
            priority = 1

        self.tree.add(priority, transition)

    # 0 ~ priorityの合計値の間でbatch sizeの分だけ乱数を生成し、
    # それに合致するデータを取得する
    def sample(self, size):
        list = []
        indexes = []
        for rand in np.random.uniform(0, self.tree.total(), size):
            (idx, _, data) = self.tree.get(rand)
            list.append(data)
            indexes.append(idx)

        return (indexes, list)

    # 再生した経験のpriorityを更新
    def update(self, idx, td_error):
        priority = self._getPriority(td_error)
        self.tree.update(idx, priority)

    def __len__(self):
        return self.size

