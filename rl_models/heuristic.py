# rl_models/heuristic.py
class HeuristicController:
    def select_action(self, state_vector):
        # state_vector mappa i raggi: [-90, -45, -22.5, 0, 22.5, 45, 90]
        left_scan = state_vector[0] + state_vector[1] + state_vector[2]
        right_scan = state_vector[4] + state_vector[5] + state_vector[6]
        
        if left_scan > right_scan + 15.0:
            return -1  # Sterza a sinistra
        elif right_scan > left_scan + 15.0:
            return 1   # Sterza a destra
        return 0       # Rettilineo