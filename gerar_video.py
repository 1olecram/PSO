import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Particle import Particle

def rastrigin(x, n, bounds):
    """Função objetivo Rastrigin para a avaliação das partículas."""
    x = np.atleast_1d(x)
    x = np.clip(x, bounds[0], bounds[1])
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)

def rastrigin_2d(x, y):
    """Função Rastrigin separada apenas para desenhar o fundo do gráfico 2D."""
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def gerar_videos_pso():
    n = 2
    bounds = (-5.12, 5.12)
    num_particles = 30
    iter_max = 100

    # A lista de parâmetros que o professor pediu:
    pares_parametros = [[0, 1], [1, 0], [1, 1], [1, 0.1], [2, 2], [4, 4]]

    for c1, c2 in pares_parametros:
        print(f"\n--- A simular C1={c1}, C2={c2} ---")
        swarm = [Particle(n, bounds) for _ in range(num_particles)]
        global_best_position = np.zeros(n)
        global_best_score = float('inf')
        history = []

        for it in range(iter_max):
            posicoes_atuais = []
            for particle in swarm:
                score = rastrigin(particle.position, n, bounds)
                particle.score = score
                
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = np.copy(particle.position)
                    
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = np.copy(particle.position)
                    
                posicoes_atuais.append(np.copy(particle.position))
                
            history.append(posicoes_atuais)

            for particle in swarm:
                # Chama a tua função que tem a lógica de atualização
                particle.update_velocity(global_best_position, c1, c2,w = 0)
                particle.update_position(bounds)

        # Geração do Gráfico e Animação
        fig, ax = plt.subplots(figsize=(8, 6))
        X = np.linspace(bounds[0], bounds[1], 100)
        Y = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(X, Y)
        Z = rastrigin_2d(X, Y)
        cp = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
        fig.colorbar(cp)
        
        ax.plot(0, 0, 'r*', markersize=15, label="Mínimo Global [0,0]")
        scatter = ax.scatter([], [], c='white', edgecolors='black', s=50, label="Partículas")
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[0], bounds[1])
        ax.legend(loc = 'upper right')

        def update(frame_number):
            posicoes = history[frame_number]
            scatter.set_offsets(posicoes)
            ax.set_title(f"Iteração: {frame_number + 1}/100 | C1={c1}, C2={c2}")
            return scatter,

        anim = animation.FuncAnimation(fig, update, frames=iter_max, interval=100, blit=True)
        nome_ficheiro = f"video_pso_c1_{c1}_c2_{c2}.gif"
        anim.save(nome_ficheiro, writer='pillow', fps=10)
        plt.close(fig) # Fecha a figura para não consumir memória
        print(f"✅ Vídeo guardado: {nome_ficheiro}")

if __name__ == "__main__":
    gerar_videos_pso()