# **🖐️ Niveau 1 : Contrôle par Gestes des Mains**

Ce module remplace le contrôleur clavier basique (`control_module.py`) par une interface de vision par ordinateur. Il utilise **MediaPipe Hands** pour suivre vos mains en temps réel via la webcam et envoyer les commandes au jeu Space Invaders.

## 🛠️ Prérequis

En plus de **Node.js** (pour le serveur WebSocket), vous aurez besoin des bibliothèques Python suivantes :
```bash
pip install websockets opencv-python mediapipe numpy
```

## **🎮 Commandes (Gestes)**

Le système détecte la position et la forme de vos mains pour envoyer des commandes au jeu :
  - **Gauche / Droite (LEFT / RIGHT)** : Placez votre main dans la moitié gauche ou droite de l'écran.
  - **Tirer (FIRE)** : Levez uniquement l'index.
  - **Valider / Start (ENTER)** : Fermez le poing.

## 🚀 Comment lancer le jeu

Pour jouer, vous devez lancer trois éléments en parallèle :
1. **Le jeu (Navigateur)** : Dans le dossier racine du projet, lancez le serveur HTTP :
```bash
python -m http.server 8000
```
Puis ouvrez http://localhost:8000 dans votre navigateur.

2. **Le pont Websocket** : Dans un autre terminal, lancez le serveur Node.js :
```bash
node server.js
```

3. **Le Contrôleur de Vision (Niveau 1) : Lancez ce script Python :
```bash
python level1_hands.py
```
