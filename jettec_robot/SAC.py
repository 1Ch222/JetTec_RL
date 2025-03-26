import numpy as np
import math
import torch
import gym
from gym import spaces
from stable_baselines3 import SAC  # Soft Actor-Critic

class ProgressionSAC(gym.Env):
    def __init__(self):
        super(ProgressionSAC, self).__init__()

        # Action space : (vitesse linéaire, vitesse angulaire)
        self.action_space = spaces.Box(low=np.array([0, -np.pi/6]), high=np.array([1, np.pi/6]), dtype=np.float32)
        
        # Observation space : (x, y, angle_theta, distance_min, distance_travelled)
        self.observation_space = spaces.Box(low=np.array([0, 0, -np.pi, 0, 0]), high=np.array([10, 10, np.pi, 10, 10]), dtype=np.float32)

        # Variables internes
        self.previous_position = None  # Position précédente pour la distance
        self.previous_angle = None     # Dernier angle pour la récompense d'orientation
        self.previous_closest_point = None  # Dernier point le plus proche de la ligne
        self.distance_travelled = 0.0  # Distance parcourue le long de la ligne
        self.alpha1 = 1.0  # Facteur pour la récompense d'orientation
        self.alpha2 = 1.0  # Facteur pour la récompense de proximité
        self.alpha3 = 1.0  # Facteur pour la récompense de progression
        self.done = False  # Indicateur si la ligne est perdue ou si la simulation est terminée

    def reset(self):
        """
        Réinitialise l'environnement. Ici, on définit l'état initial.
        """
        self.previous_position = (0, 0)  # Position initiale
        self.previous_angle = 0  # Angle initial
        self.previous_closest_point = None
        self.distance_travelled = 0.0
        self.done = False

        return np.array([0, 0, 0, 10, 0])  # Renvoie un état initial fictif

    def step(self, action):
        """
        Effectue une étape de l'environnement avec l'action donnée (vitesse linéaire, vitesse angulaire).
        """
        v, w = action  # Action : v vitesse linéaire, w vitesse angulaire

        # Simulation de la progression du robot, mettre à jour sa position et son angle
        # Note : Tu dois ajouter la logique de déplacement du robot en fonction de l'action.
        # Pour simplifier, on fait une simple mise à jour ici.
        new_position = (self.previous_position[0] + v, self.previous_position[1] + w)  # Mise à jour fictive

        # Calculer l'angle theta, distance min, et la distance parcourue entre deux itérations
        angle_theta = self.calculate_theta(new_position)
        closest_point, closest_distance = self.calculate_closest_point(new_position)  # Calculer le point le plus proche
        distance = self.calculate_distance(self.previous_position, new_position)

        # Calculer les récompenses
        r1 = self.calculate_r1(angle_theta)
        r2 = self.calculate_r2(closest_distance)
        r3 = self.alpha3 * distance  # Récompense de progression
        reward = r1 + r2 + r3  # Récompense totale

        # Si la ligne est perdue (pas de détection de centroïde), ajouter -1 à la récompense
        if closest_point is None:
            reward = -1.0
            self.done = True

        # Mettre à jour la position et l'angle précédent pour la prochaine itération
        self.previous_position = new_position
        self.previous_angle = angle_theta

        return np.array([new_position[0], new_position[1], angle_theta, closest_distance, self.distance_travelled]), reward, self.done, {}

    def calculate_theta(self, new_position):
        """
        Calcule l'angle (theta) entre la direction du robot et la ligne à suivre.
        """
        # Ici, tu dois calculer theta, basé sur la direction du robot et la ligne. 
        # Pour simplifier, on prend un angle fixe pour l'exemple.
        return np.random.uniform(-np.pi, np.pi)  # Un angle aléatoire entre -π et π

    def calculate_closest_point(self, new_position):
        """
        Trouve le point de la ligne le plus proche du robot.
        """
        # Pour l'exemple, on utilise une distance aléatoire.
        closest_point = (new_position[0], new_position[1])
        closest_distance = np.random.uniform(0, 10)  # Valeur aléatoire pour la distance
        return closest_point, closest_distance

    def calculate_distance(self, prev_pos, current_pos):
        """
        Calcule la distance entre deux positions données.
        """
        return np.sqrt((current_pos[0] - prev_pos[0]) ** 2 + (current_pos[1] - prev_pos[1]) ** 2)

    def calculate_r1(self, angle_theta):
        """
        Calcule la récompense d'orientation en fonction de l'angle (r1).
        """
        return self.alpha1 * math.cos(angle_theta)  # Utilise le cos de l'angle pour l'orientation

    def calculate_r2(self, closest_distance):
        """
        Calcule la récompense de proximité de la ligne en fonction de la distance au point le plus proche (r2).
        """
        max_distance = 10.0  # Distance maximale possible (à ajuster)
        return self.alpha2 * (closest_distance / max_distance)

# Entraînement du modèle SAC
def main():
    env = ProgressionSAC()  # Crée l'environnement
    model = SAC('MlpPolicy', env, verbose=1)  # Crée un modèle SAC avec une politique MLP (Multi-Layer Perceptron)
    
    # Entraînement du modèle SAC
    model.learn(total_timesteps=10000)

if __name__ == "__main__":
    main()

