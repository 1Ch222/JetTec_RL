import random

class RLModule:
    def __init__(self):
        self.q_table = {}  # Q-table pour Q-learning
        self.alpha = 0.1   # Taux d'apprentissage
        self.gamma = 0.9   # Facteur de discount
        self.epsilon = 0.1 # Taux d'exploration (epsilon-greedy)
        
    def choose_action(self, state):
        """
        Choisit une action basée sur la politique epsilon-greedy.
        """
        if random.random() < self.epsilon:
            # Exploration : Choisir une action aléatoire
            return random.choice([0, 1, 2])  # 0 = tourner à gauche, 1 = avancer, 2 = tourner à droite
        else:
            # Exploitation : Choisir la meilleure action basée sur la Q-table
            if state not in self.q_table:
                self.q_table[state] = [0, 0, 0]  # Initialiser si l'état n'existe pas
            return np.argmax(self.q_table[state])  # Choisir l'action avec la plus grande valeur Q

    def execute_action(self, action, pid):
        """
        Exécute l'action en utilisant le contrôleur PID.
        """
        if action == 0:  # Tourner à gauche
            pid.setpoint -= 10  # Ajuster le setpoint du PID pour tourner à gauche
        elif action == 1:  # Avancer
            pid.setpoint = 0  # Ajuster le setpoint du PID pour avancer tout droit
        elif action == 2:  # Tourner à droite
            pid.setpoint += 10  # Ajuster le setpoint du PID pour tourner à droite

        # Calculer la sortie PID
        pid_output = pid.update(current_value=pid.setpoint)
        
        # Appliquer la sortie PID pour ajuster le mouvement (par exemple, ajuster la vitesse ou l'angle)
        # Par exemple, tu pourrais contrôler la vitesse ou l'orientation du robot ici.
        return pid_output

