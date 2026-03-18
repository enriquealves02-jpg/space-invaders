# **🎾 Niveau 3 : Suivi d'Objet par Couleur

Ce module vous permet d'utiliser n'importe quel objet coloré (balle, stylo, téléphone) comme manette pour Space Invaders. Il remplace le fichier `control_module.py` et utilise **OpenCV** pour suivre l'objet en espace HSV.

## 🛠️ Prérequis

**Node.js** est requis pour le serveur relais. Côté Python :
```bash
pip install websockets opencv-python numpy
```

## **🎯 Commandes et Calibration**

Par défaut, l'algorithme cible une couleur orange. Pour utiliser votre propre objet :
  1. Placez l'objet dans la croix au centre de la caméra.
  2. Appuyez sur la touche c pour calibrer la nouvelle couleur.

**Contrôle via la position de l'objet** :
  - **Moitié Gauche / Droite** : Déplace le vaisseau (LEFT / RIGHT).
  - **Tiers Supérieur (Haut)** : Déclenche un tir (FIRE).
  - **Tiers Inférieur (Bas)** : Envoie la commande pour démarrer (ENTER).

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
3. **Le Contrôleur Couleur (Niveau 3) : Lancez ce script Python :
```bash
python level3_color.py
```
Appuyez sur q pour fermer la fenêtre vidéo.
