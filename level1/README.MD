# 🖐️ Niveau 1 : Contrôle par Gestes des Mains

Ce module remplace le contrôleur clavier basique (`control_module.py`) par une interface de vision par ordinateur. Il utilise **MediaPipe Hands** pour suivre vos mains en temps réel via la webcam et envoyer les commandes au jeu Space Invaders.

## 🛠️ Prérequis

En plus de **Node.js** (pour le serveur WebSocket), vous aurez besoin des bibliothèques Python suivantes :
```bash
pip install websockets opencv-python mediapipe numpy
