# team.py
from config import TEAM_REWARD_SHARE


class Team:
    def __init__(self, team_id, algorithm_name, cars_list):
        self.team_id = team_id
        self.algorithm_name = algorithm_name
        self.cars = cars_list  

    def compute_cooperative_rewards(self, individual_rewards_dict, cars_alive_dict):
        rewards = {}
        id1, id2 = self.cars[0].car_id, self.cars[1].car_id
        
        r1 = individual_rewards_dict.get(id1, 0.0)
        r2 = individual_rewards_dict.get(id2, 0.0)

        if cars_alive_dict[id1] and cars_alive_dict[id2]:
            rewards[id1] = r1 + TEAM_REWARD_SHARE * max(0.0, r2)
            rewards[id2] = r2 + TEAM_REWARD_SHARE * max(0.0, r1)
        else:
            rewards[id1] = r1
            rewards[id2] = r2

        return rewards