class PID:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp  # Coefficient proportionnel
        self.ki = ki  # Coefficient intégral
        self.kd = kd  # Coefficient dérivé
        self.setpoint = setpoint  # Valeur cible (setpoint)
        
        self.previous_error = 0  # Erreur précédente
        self.integral = 0  # Terme intégral
        self.output = 0  # Sortie PID

    def update(self, current_value):
        """
        Met à jour le calcul PID en fonction de l'erreur actuelle.
        """
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.previous_error
        self.output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # Sauvegarder l'erreur pour le calcul du terme dérivé au prochain cycle
        self.previous_error = error
        
        return self.output

