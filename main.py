from Particle import Particle
import numpy as np

def rastrigin(x, n, bounds):
    """
    Avalia a função Rastrigin para um array x de n dimensões.
    O mínimo global está em x = [0, 0, ..., 0] com um valor de 0.
    """
    x = np.atleast_1d(x)
    x = np.clip(x, bounds[0], bounds[1])
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)

def main():

    n = 2 # Dimensões do problema
    bounds = (-5.12, 5.12) # Limites do problema
    c1 = 2
    c2 = 2
    
    
    print(f"Preparando PSO para a função Rastrigin com n={n} e limites {bounds}")
        
    # Inicializando o Enxame (Swarm)
    num_particles = 30
    swarm = [Particle(n, bounds) for _ in range(num_particles)]
    
    print(f"\n--- Estrutura do PSO ---")
    print(f"Enxame criado com {len(swarm)} partículas.")
    print(f"Partícula 3 - Posição inicial: {swarm[3].position}")
    print(f"Partícula 3 - Velocidade inicial: {swarm[3].velocity}")

    # TODO: Implementar o laço de evolução do PSO (avaliação, atualização de pbest/gbest e movimentação)

if __name__ == "__main__":
    main()