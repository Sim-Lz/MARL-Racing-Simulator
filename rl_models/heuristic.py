# rl_models/heuristic.py
class HeuristicController:
    def select_action(self, state_vector):
        # Esponiamo l'attributo per uniformità strutturale con gli altri modelli
        if not hasattr(self, 'last_grad_norm'):
            self.last_grad_norm = 0.0

        left_scan = state_vector[0] + state_vector[1] + state_vector[2]
        right_scan = state_vector[4] + state_vector[5] + state_vector[6]
        center_scan = state_vector[3]
        
        # Determina la componente dello sterzo
        if left_scan > right_scan + 15.0:
            steer = -1  # Sinistra
        elif right_scan > left_scan + 15.0:
            steer = 1   # Destra
        else:
            steer = 0   # Dritto

        # Determina la componente longitudinale basandosi sullo spazio frontale
        if center_scan > 160.0:
            longitudinal = 1   # Accelerazione massima
        elif center_scan < 90.0:
            longitudinal = -1  # Frenata protettiva
        else:
            longitudinal = 0   # Mantenimento per inerzia

        # Dizionario inverso di mappatura per ricavare l'azione corretta da 0 a 8
        # Struttura: (steer_act, long_act) -> azione discreta
        reverse_map = {
            (-1,  1): 0, (0,  1): 1, (1,  1): 2,
            (-1,  0): 3, (0,  0): 4, (1,  0): 5,
            (-1, -1): 6, (0, -1): 7, (1, -1): 8
        }
        return reverse_map.get((steer, longitudinal), 1)