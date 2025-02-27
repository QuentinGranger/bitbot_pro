# BitBot Pro Data

## Structure des données

- raw/ : Données brutes des marchés et des API
- processed/ : Données nettoyées et transformées
- models/ : Modèles entraînés et checkpoints

## Utilisation avec DVC

Pour récupérer les données :
```bash
dvc pull
```

Pour ajouter de nouvelles données :
```bash
dvc add data/raw/nouveau_dataset.csv
git add data/raw/nouveau_dataset.csv.dvc
git commit -m 'Ajout nouveau dataset'
dvc push
```
