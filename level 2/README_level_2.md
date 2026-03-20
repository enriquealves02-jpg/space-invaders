# **🧍 Niveau 2 : Contrôle par Pose Corporelle**

Ce module remplace le contrôleur classique par une expérience physique ! Il utilise **MediaPipe Pose** pour analyser l'inclinaison de votre corps et la position de vos bras afin de contrôler le vaisseau de Space Invaders.

## 🛠️ Prérequis

Assurez-vous d'avoir **Node.js** installé, ainsi que les dépendances Python suivantes :
```bash
pip install websockets opencv-python mediapipe numpy
```

## **🎮 Commandes (Corps)**

  - **Gauche / Droite (LEFT / RIGHT)** : Penchez vos épaules vers la gauche ou vers la droite. Une jauge à l'écran vous aide à jauger l'inclinaison.
  - **Tirer (FIRE)** : Levez un bras (poignet au-dessus de l'épaule).
  - **Valider / Start (ENTER)** : Cachez vos mains (dans votre dos ou hors du cadre de la caméra).

## 🚀 Comment lancer le jeu

Vous devez faire tourner l'architecture complète :
1. **Le jeu (Navigateur)** : Dans le dossier racine du projet, lancez le serveur HTTP :
```bash
python -m http.server 8000
```
Puis ouvrez http://localhost:8000 dans votre navigateur.

2. **Le pont Websocket** : Dans un autre terminal, lancez le serveur Node.js :
```bash
node server.js
```
3. **Le Contrôleur Pose (Niveau 2) : Lancez ce script Python :
```bash
python level2_pose.py
```
Reculez un peu pour que la caméra capte le haut de votre corps. Appuyez sur q sur la fenêtre de la caméra pour quitter.