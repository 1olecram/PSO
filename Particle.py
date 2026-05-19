import numpy as np

class Particle:
    """
    Classe que representa uma partícula no enxame do PSO.
    """
    def __init__(self, n, bounds):
        # Posição inicial aleatória dentro dos limites
        self.position = np.random.uniform(bounds[0], bounds[1], n)
        
        # Velocidade inicial aleatória (pode variar entre os limites)
        self.velocity = np.random.uniform(bounds[0], bounds[1], n)
        
        # Memória da melhor posição individual (pbest)
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')  # Minimização: inicializa com infinito
        
        # Pontuação atual
        self.score = float('inf')

    def update_velocity(self, local_best_position, c1, c2, w):
        cognitive_component = c1 * np.random.uniform(0, 1, len(self.position)) * (self.best_position - self.position)
        social_component = c2 * np.random.uniform(0, 1) * (local_best_position - self.position)
        # calculo da velocidade usando o fator de constrição, que nao possui o parametro w(inercia)
        self.velocity = self.constriction_factor(c1, c2) * (self.velocity + cognitive_component + social_component)
    
    def update_position(self, bounds):
        self.position = self.position + self.velocity
        
        # Máscara booleana para identificar as dimensões que ultrapassaram os limites
        out_of_bounds = (self.position < bounds[0]) | (self.position > bounds[1])
        
        # Limita as posições dentro dos limites do problema
        self.position = np.clip(self.position, bounds[0], bounds[1]) 
        
        # Zera a velocidade apenas nas dimensões que bateram nos limites
        self.velocity[out_of_bounds] = 0.0
        
    def constriction_factor(self, c1, c2):
        """
        Calcula o fator de constrição para o PSO.
        """
        phi = c1 + c2
        if phi <= 4:
            return 1.0 # Evita raiz negativa se phi <= 4 acidentalmente
        return 2 / np.abs(2 - phi - np.sqrt(phi**2 - 4 * phi))