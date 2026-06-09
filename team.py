# team.py
class Team:
    def __init__(self, team_id, algorithm_name, cars_list):
        self.team_id = team_id
        self.algorithm_name = algorithm_name
        self.cars = cars_list  # Array contenente esattamente 2 istanze di Car

    def compute_cooperative_rewards(self):
        """Calcola e bilancia i reward cooperativi basandosi sulle performance istantanee."""
        rewards = {}
        for car in self.cars:
            if not car.alive:
                rewards[car.car_id] = -15.0  # Penalità forte per morte prematura
            else:
                # Ricompensa proporzionale alla stabilità della velocità espressa
                rewards[car.car_id] = car.speed * 1.5  
        
        # Blending cooperativo dei canali di comunicazione (70% individuale, 30% alleato)
        if len(self.cars) == 2:
            id1, id2 = self.cars[0].car_id, self.cars[1].car_id
            r1 = rewards[id1]
            r2 = rewards[id2]
            
            rewards[id1] = 0.7 * r1 + 0.3 * r2
            rewards[id2] = 0.7 * r2 + 0.3 * r1
            
        return rewards