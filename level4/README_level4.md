# Space Invaders - Agent IA par Vision par Ordinateur (Niveau 4)

Agent DQN (Deep Q-Network) qui apprend a jouer a Space Invaders en analysant les pixels de l'ecran avec un CNN (reseau de neurones convolutif). Entraine pendant 4 jours (~6860 episodes, ~2.15M steps) sur GPU NVIDIA RTX 5060.

## Prerequis

- **Node.js** (v18+) : [nodejs.org](https://nodejs.org/)
- **Python 3.10+**
- **GPU NVIDIA avec CUDA** (recommande pour l'entrainement, pas necessaire pour tester)

## Installation

```bash
# 1. Installer les dependances Node.js
npm install

# 2. Installer les dependances Python
pip install -r requirements.txt
```

> **Note PyTorch GPU** : Si vous avez un GPU NVIDIA, installez PyTorch avec CUDA :
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

## Tester l'agent IA (ai_play.py)

L'agent entraine est sauvegarde dans `dqn_model.pth`. Pour le voir jouer :

**Ouvrez 3 terminaux dans ce dossier :**

### Terminal 1 : Serveur WebSocket
```bash
node server.js
```

### Terminal 2 : Serveur HTTP (le jeu)
```bash
python -m http.server 8000
```

### Terminal 3 : Ouvrez le jeu puis lancez l'agent
1. Ouvrez **http://localhost:8000** dans votre navigateur
2. Lancez l'agent :
```bash
python ai_play.py
```
3. Une fenetre s'ouvre : **dessinez un rectangle autour du jeu** avec la souris, puis appuyez sur ENTREE
4. L'agent joue automatiquement. Appuyez sur Ctrl+C pour arreter.

## Tester le controle par gestes (cv_control_module.py)

Controle du jeu avec la main via webcam (MediaPipe) :

```bash
python cv_control_module.py
```

- Main a gauche → GAUCHE
- Main a droite → DROITE
- Index leve → TIR
- Touche 'e' dans la fenetre camera → ENTER (demarrer le jeu)
- Touche 'q' → quitter

## Relancer l'entrainement (ai_agent.py)

Pour reprendre l'entrainement du modele existant :

```bash
python ai_agent.py
```

L'agent charge automatiquement `dqn_model.pth` et reprend ou il en etait. Le modele est sauvegarde tous les 50 episodes. Ctrl+C pour arreter (sauvegarde automatique).

## Structure des fichiers

| Fichier | Description |
|---------|-------------|
| `ai_agent.py` | Agent DQN : CNN + entrainement par renforcement |
| `ai_play.py` | Mode lecture : charge le modele entraine et joue |
| `cv_control_module.py` | Controle par gestes (webcam + MediaPipe) |
| `control_module.py` | Controle clavier basique (module original) |
| `dqn_model.pth` | Modele CNN entraine (~6860 episodes) |
| `server.js` | Serveur WebSocket (pont entre agent et jeu) |
| `game.bundle.js` | Code du jeu (modifie pour envoyer l'etat via WebSocket) |
| `index.html` | Page du jeu |
| `requirements.txt` | Dependances Python |

## Architecture technique

```
Navigateur (jeu)  <-- mss (capture ecran) -->  ai_agent.py (CNN + DQN)
                   <-- WebSocket (port 8765) -->  GPU CUDA
                   commandes + etat du jeu
```

- **CNN** : 3 couches Conv2d + 2 couches Linear (architecture DQN Atari)
- **Input** : 4 frames grayscale 84x84 empilees
- **Output** : Q-values pour 4 actions (RIEN, GAUCHE, DROITE, TIR)
- **Reward** : +100/kill, -1000/vie perdue, +5000/victoire

## Licence

MIT License
