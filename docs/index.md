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
- Leo Tremblay (20247961): Chercheur

## Description du projet

### Contexte

Ce projet touche les domaines de science de données, apprentissage automatique (Machine Learning), apprentissage profond (Deep Learning), apprentissage par renforcement (Reinforcement Learning) ainsi que la robotique.  Dans le domaine de la robotique, ces différents types d'apprentissage sont très important afin de permette aux robots de s'adapter à différents scénarios.  En effet, il est très difficile sinon impossible de trouver des solutions déterministes aux situations de la vie courante.  Les robots doivent être capable de s'adapter à de nouvelles situations et c'est pour ces cas particuliers que l'apprentissage automatique et ses différents dérivés entrent en jeu.  Les récentes recherches montrent que l’utilisation de l’apprentissage profond ainsi que l’apprentissage par renforcement dans la robotique permettent d’accomplir plus que les anciennes méthodes déterministes ou les méthodes faisant usage de réseaux neuronaux peu profonds.  Cependant, le transfert de la simulation vers des environnements réels est toujours un défi puisque ces deux environnements peuvent être très différents en pratique.  Ces techniques d’apprentissage ainsi que le transfert de la simulation au réel (sim2real) sont des éléments clés au développement de robots pouvant être utilisés au quotidien.  En effet, cette réalité de côtoyer des robots au quotidien est de plus en plus partagée par tous.  En partant de l’automatisation de différents secteurs jusqu’à la surveillance ou même pour la livraison, les robots sont de plus en plus présents dans nos vies.  Enfin, la dernière considération pour notre projet est celle de la généralisation en apprentissage.  Ceci est un besoin grandissant dans le domaine de la robotique afin de rendre des robots ou des modèles plus polyvalents.

### Problématique ou motivations

 Ce projet nous permettra d’apprendre à bien faire le transfert de modèles entrainés dans un environnement simulé vers un robot réel.  Comme mentionné précédemment, cette transition n’est pas toujours facile à faire puisque les deux environnements sont différents ce qui fait en sorte que le robot n’agit pas toujours comme anticipé lorsqu’on fait le transfert.  Bien que la simulation existe depuis déjà plusieurs décennies, elle est une nécessité dans le domaine de la robotique de nos jours.  La simulation permet d’entrainer des modèles sur de plus grands ensembles de données, mais de façon potentiellement plus rapide, moins cher et en réduisant les risques ([MIT-Sim2Real_T-ASE.pdf](https://dspace.mit.edu/bitstream/handle/1721.1/138850/2021-04-Sim2Real_T-ASE.pdf?sequence=2)).  Il est donc crucial de savoir comment bien faire ce transfert en deux environnements nécessaires à la robotique.  De plus, ce projet nous apprendra également sur la généralisation en apprentissage.  Cette pratique permet de généraliser les modèles afin de les permettre de performer sur des robots de différents types sans connaitre tous les détails de l’hôte.  Cette technique est très utile et rend les modèles beaucoup plus polyvalents, mais la recette n’est pas donnée.  Il faut réussir à adapter le modèle sans savoir si les joints, les moteurs ou les différentes parties du robots seront les mêmes, on doit donc inférer des données et trouver des techniques permettant d’anticiper les variations sur le robot.

### Proposition et objectifs
Au tout début du projet, notre superviseur Glen Berseth nous a donné des objectifs à compléter avant la fin de chaque mois pour que nous puissions travailler d’une manière structurée et organisée. Pour le mois de mai, l’objectif est de nous renseigner sur l’apprentissage par renforcement et sur les technologies qui seront utilisées le long du projet, ainsi que de préparer l’environnement de travail pour nous permettre à ensuite entraîner des modèles dans une simulation. En juin, nous allons d’abord entraîner les robots dans une simulation à faire certaines tâches simples telles que de se tourner en rond ou de se naviguer autour des obstacles, et puis tenter de transférer notre modèle entraîné à un robot dans la vraie vie pour voir si le vrai robot peut accomplir les tâches aussi bien que les robots simulés. En juillet nous prévoyons finalement commencer à généraliser le modèle d’apprentissage pour qu’il puisse être utilisé sur des robots avec des formes différentes. La généralisation sera d’abord faite dans la simulation avant de passer sur des robots physiques. Pour accomplir cela nous allons devoir trouver une manière de permettre notre modèle à apprendre sur la morphologie (nombre de pattes, etc.) d’un robot donné pour qu’il puisse ensuite contrôler le robot efficacement. En août nous allons finaliser notre recherche et écrire un rapport pour le projet.

### Méthodologie
Il n’y a pas de méthodologie de recherche établie encore pour notre projet, mais nous avons pourtant une idée générale de comment l’entraînement et les tests des modèles d’apprentissage seront fait. Tout apprentissage sera fait à travers le logiciel de simulation car il n’y aura aucun risque d’endommagement des robots physiques et cela nous permettrait également de paralléliser l’apprentissage, c’est-à-dire le modèle peut être entraîné sur des dizaines ou des centaines de robots en même temps, ce qui accélère grandement le processus d’apprentissage. Lorsque nous avons vérifié que le modèle marche bien dans la simulation, le transfert aux robots physiques peut être effectué. Il est possible que le modèle ne performe pas aussi bien dans la vraie vie que dans la simulation, et dans ce cas là il sera nécessaire de revenir dans la simulation pour faire des améliorations. Pour accomplir la généralisation de nos modèles, il sera nécessaire de trouver soit un encodage universel pour la morphologie des robots ou potentiellement une manière d’incorporer l’apprentissage automatique pour que le modèle soit capable d’apprendre sur la morphologie du robot au fur et à mesure qu’il le contrôle. Les détails concernant ce sujet n’est toujours pas clair et des nombreux essais seront nécessaires afin de réaliser un bon niveau de généralisation.

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
