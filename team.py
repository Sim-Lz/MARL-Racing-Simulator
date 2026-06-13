# team.py
class Team:
    def __init__(self, team_id, algorithm_name, cars_list):
        self.team_id = team_id
        self.algorithm_name = algorithm_name
        self.cars = cars_list  

    def compute_cooperative_rewards(self, individual_rewards_dict, cars_alive_dict):
        """
        Riceve i reward totali pre-calcolati individualmente ed esegue il blending 90/10.
        Se un'auto è eliminata, scherma l'auto superstite per impedirne il crollo valutativo.
        """
        rewards = {}
        id1, id2 = self.cars[0].car_id, self.cars[1].car_id
        
        r1 = individual_rewards_dict.get(id1, 0.0)
        r2 = individual_rewards_dict.get(id2, 0.0)
        
        # Protezione anti-morte del compagno (Risoluzione Bug di Trascinamento Negativo)
        if not cars_alive_dict[id1] and cars_alive_dict[id2]:
            rewards[id1] = r1
            rewards[id2] = r2  # L'auto superstite si focalizza al 100% sul proprio target
        elif not cars_alive_dict[id2] and cars_alive_dict[id1]:
            rewards[id1] = r1
            rewards[id2] = r2
        else:
            # Mantenimento del Blending 90% Individuale / 10% Cooperativo richiesto
            rewards[id1] = 0.9 * r1 + 0.1 * r2
            rewards[id2] = 0.9 * r2 + 0.1 * r1
            
        return rewards