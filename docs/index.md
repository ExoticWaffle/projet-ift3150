# Projet IFT3150: Nom du projet

> **Thèmes**: Science de données, Machine Learning, Deep Learning, Reinforment Learning, Robotique  
> **Superviseur**: Glen Berseth  

## Informations importantes

!!! info "Dates importantes"
    - **Description du projet** : 16 mai 2025
    - **Foire 1: Prototypage** : 9-13 juin 2025
    - **Foire 2: Version beta** : 14-18 juillet 2025
    - **Présentation et rapport** : 11-15 août 2025

## Équipe

- Jacob Denault (20296116): Chercheur
- Leo Tremblay: Chercheur

## Description du projet

### Contexte

Ce projet touche les domaines de science de données, apprentissage automatique (Machine Learning), apprentissage profond (Deep Learning), apprentissage par renforcement (Reinforcement Learning) ainsi que la robotique.  Dans le domaine de la robotique, ces différents types d'apprentissage sont très important afin de permette aux robots de s'adapter à différents scénarios.  En effet, il est très difficile sinon impossible de trouver des solutions déterministes aux situations de la vie courante.  Les robots doivent être capable de s'adapter à de nouvelles situations et c'est pour ces cas particuliers que l'apprentissage automatique et ses différents dérivés entrent en jeu.  Les récentes recherches montrent que l’utilisation de l’apprentissage profond ainsi que l’apprentissage par renforcement dans la robotique permettent d’accomplir plus que les anciennes méthodes déterministes ou les méthodes faisant usage de réseaux neuronaux peu profonds.  Cependant, le transfert de la simulation vers des environnements réels est toujours un défi puisque ces deux environnements peuvent être très différents en pratique.  Ces techniques d’apprentissage ainsi que le transfert de la simulation au réel (sim2real) sont des éléments clés au développement de robots pouvant être utilisés au quotidien.  En effet, cette réalité de côtoyer des robots au quotidien est de plus en plus partagée par tous.  En partant de l’automatisation de différents secteurs jusqu’à la surveillance ou même pour la livraison, les robots sont de plus en plus présents dans nos vies.  Enfin, la dernière considération pour notre projet est celle de la généralisation en apprentissage.  Ceci est un besoin grandissant dans le domaine de la robotique afin de rendre des robots ou des modèles plus polyvalents.

### Problématique ou motivations

 Ce projet nous permettra d’apprendre à bien faire le transfert de modèles entrainés dans un environnement simulé vers un robot réel.  Comme mentionné précédemment, cette transition n’est pas toujours facile à faire puisque les deux environnements sont différents ce qui fait en sorte que le robot n’agit pas toujours comme anticipé lorsqu’on fait le transfert.  Bien que la simulation existe depuis déjà plusieurs décennies, elle est une nécessité dans le domaine de la robotique de nos jours.  La simulation permet d’entrainer des modèles sur de plus grands ensembles de données, mais de façon potentiellement plus rapide, moins cher et en réduisant les risques ([MIT-Sim2Real_T-ASE.pdf](https://dspace.mit.edu/bitstream/handle/1721.1/138850/2021-04-Sim2Real_T-ASE.pdf?sequence=2)).  Il est donc crucial de savoir comment bien faire ce transfert en deux environnements nécessaires à la robotique.  De plus, ce projet nous apprendra également sur la généralisation en apprentissage.  Cette pratique permet de généraliser les modèles afin de les permettre de performer sur des robots de différents types sans connaitre tous les détails de l’hôte.  Cette technique est très utile et rend les modèles beaucoup plus polyvalents, mais la recette n’est pas donnée.  Il faut réussir à adapter le modèle sans savoir si les joints, les moteurs ou les différentes parties du robots seront les mêmes, on doit donc inférer des données et trouver des techniques permettant d’anticiper les variations sur le robot.

### Proposition et objectifs


## Échéancier

!!! info
    Le suivi complet est disponible dans la page [Suivi de projet](suivi.md).

| Jalon (*Milestone*)            | Date prévue   | Livrable                            | Statut      |
|--------------------------------|---------------|-------------------------------------|-------------|
| Ouverture de projet            | 1 mai         | Proposition de projet               | ✅ Terminé  |
| Analyse des exigences          | 16 mai        | Document d'analyse                  | 🔄 En cours |
| Prototype 1                    | 23 mai        | Maquette + Flux d'activités         | ⏳ À venir  |
| Prototype 2                    | 30 mai        | Prototype finale + Flux             | ⏳ À venir  |
| Architecture                   | 30 mai        | Diagramme UML ou modèle C4          | ⏳ À venir  |
| Modèle de donneés              | 6 juin        | Diagramme UML ou entité-association | ⏳ À venir  |
| Revue de conception            | 6 juin        | Feedback encadrant + ajustements    | ⏳ À venir  |
| Implémentation v1              | 20 juin       | Application v1                      | ⏳ À venir  |
| Implémentation v2 + tests      | 11 juillet    | Application v2 + Tests              | ⏳ À venir  |
| Implémentation v3              | 1er août      | Version finale                      | ⏳ À venir  |
| Tests                          | 11-31 juillet | Plan + Résultats intermédiaires     | ⏳ À venir  |
| Évaluation finale              | 8 août        | Analyse des résultats + Discussion  | ⏳ À venir  |
| Présentation + Rapport         | 15 août       | Présentation + Rapport              | ⏳ À venir  |
