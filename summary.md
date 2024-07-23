## Class

### Input JSON

La classe prend en input un JSON. Par exemple :

```json
{
  "w_whale": [0.1, 0.05, 0.13, 0.11, 0.02, 0.04, 0.08],
  "phwh": [0.6, 0.5, 0.4, 0.55, 0.44, 0.51, 0.4],
  "rhwh": [1.2, 0.8, 1.1, 1.3, 0.5, 0.9, 0.5],
  "rbwh": [-0.8, -1, -0.8, -1.1, -1.1, -0.9, -0.5],
  "x": [0.02, 0.05, 0.04, 0.03, 0.02, 0.04, 0.01],
  "y": [0.02, 0.05, 0.04, 0.03, 0.02, 0.04, 0.01],
  "ppc": 0.6,
  "phld": 0.5,
  "rhld": 1,
  "rbld": -1,
  "N_indice": 100,
  "N_ptf": 30
}
```

#### Valeurs par défaut

Si x et y ne sont pas fournis dans le JSON, ils seront initialisés à zéro.

#### Vecteurs générés automatiquement

Si les probabilités de rendements haussiers des whales (phwh) sont les mêmes, une seule valeur peut être inscrite et le vecteur sera créé automatiquement. Par exemple, si toutes les probabilités de rendements haussiers sont de $0.5$, on peut écrire :

```json
{
  "w_whale": [0.1, 0.05, 0.13, 0.11, 0.02, 0.04, 0.08],
  "phwh": 0.5,
  "rhwh": [1.2, 0.8, 1.1, 1.3, 0.5, 0.9, 0.5],
  "rbwh": [-0.8, -1, -0.8, -1.1, -1.1, -0.9, -0.5],
  "ppc": 0.6,
  "phld": 0.5,
  "rhld": 1,
  "rbld": -1,
  "N_indice": 100,
  "N_ptf": 30
}
```

De même, si les rendements haussiers (rhwh) ou baissiers (rbwh) des whales sont les mêmes, une seule valeur peut être inscrite et le vecteur sera créé automatiquement. Par exemple, pour des rendements haussiers de $1.2$ et des rendements baissiers de $-0.8$ pour toutes les whales :

```json
{
  "w_whale": [0.1, 0.05, 0.13, 0.11, 0.02, 0.04, 0.08],
  "phwh": 0.5,
  "rhwh": 1.2,
  "rbwh": -0.8,
  "ppc": 0.6,
  "phld": 0.5,
  "rhld": 1,
  "rbld": -1,
  "N_indice": 100,
  "N_ptf": 30
}
```

Cela simplifie la création des paramètres lorsque les valeurs sont homogènes. 

La seule condition à respecter est de proposer un vecteur $w_{whale}$ avec une liste de la longueur du nombre de *Whales*, même si elles ont toutes le même poids. Car $n_{whale}$ est déterminé grâce à la longueur de notre vecteur $w_{whale}$.

### Output JSON

La mission principale de notre Class est retourner les valeurs de $x$ et $y$ optimales pour les conditions que nous avons définis. Nous récupérons donc un vecteur $x$ et un vecteur $y$ de taille $n_{whale}$.

## Paramètres

### Généraux

-   $N_{indice}$ : le nombre de valeurs dans l'indice
-   $N_{portefeuille}$ : le nombre de valeurs dans le portefeuille
-   $p_{pc}$ : la probabilité de prédiction correcte

### Whales

-   $n_{whale}$ : le nombre de whales
-   $w_{whale}$ : le poids des whales (vecteur de dimension $n_{whale}$)
-   $p_{h}^{whale}$ : la probabilité de rendements haussiers associés aux whales (vecteur de dimension $n_{whale}$)
-   $r_{h}^{whale}$ : les rendements haussiers associés aux whales (vecteur de dimension $n_{whale}$)
-   $r_{b}^{whale}$ : les rendements baissiers associés aux whales (vecteur de dimension $n_{whale}$)
-   $x$ : le vecteur de poids des rendements haussiers associés aux whales (vecteur de dimension $n_{whale}$)
-   $y$ : le vecteur de poids des rendements baissiers associés aux whales (vecteur de dimension $n_{whale}$)
            
Exemple du vecteur $w_{whale}$ : 

$
w_{whale} = \begin{pmatrix}
w_{whale_1} \\
w_{whale_2} \\
\vdots \\
w_{whale_i} \\
\vdots \\
w_{whale_{n_{whale}}}
\end{pmatrix}
$


### Lambdas

-   $p_{h}^{lambda}$ : la probabilité de rendements haussiers associés aux lambdas
-   $r_{h}^{lambda}$ : les rendements haussiers associés aux lambdas
-   $r_{b}^{lambda}$ : les rendements baissiers associés aux lambdas

## Calculs
            
### Rendement de l'indice

##### $$R_{indice} = La + B$$

Où :
            
##### **$L$** : La distribution de rendement de nos *Lambdas*, soit **$N_{indice} - n_{whale}$** valeurs.

L'espérance de rendement de cette distribution est simple à calculer :

-   Nous avons une probabilité $p_{h}$ d'avoir un rendement $r_{h}$.

-   Une probabilité $1 - p_{h}$ d'avoir un rendement $r_{b}$.

Donc :

$$
E(L) = p_{h} \times r_{h} + \left( 1 - p_{h} \right) \times r_{b}
$$

Et sa variance :

$$
Var(L) = \frac{p_{h} \times r_{h}^{2} + \left( 1 - p_{h} \right) \times r_{b}^{2} - E(L)^{2}}{N_{indice} - n_{whale}}
$$

##### **$B$** : La distribution de rendement de nos *Whales*, soit **$n_{whale}$** valeurs.

L'espérance de rendement de cette distribution est tout aussi simple.
Nous avons :

-   Une probabilité $p_{h}$ d'avoir un rendement $r_{h}$ avec un poids
    de $w_{whale}$.

-   Une probabilité $1 - p_{h}$ d'avoir un rendement $r_{b}$ avec un
    poids de $w_{whale}$

Et sachant qu'on en a $n_{whale}$. Donc :

$$
E(B) = \sum_{i=1}^{n_{whale}} \left( p_{h,i}^{whale} \cdot r_{h,i}^{whale} + (1 - p_{h,i}^{whale}) \cdot r_{b,i}^{whale} \right) \cdot w_{whale_i}
$$
            
Et sa variance :

$$
Var(B) = \sum_{i=1}^{n_{whale}} \left[ \left( p_{h,i}^{whale} \cdot (r_{h,i}^{whale} \cdot w_{whale_i})^2 + (1 - p_{h,i}^{whale}) \cdot (r_{b,i}^{whale} \cdot w_{whale_i})^2 \right) \right] - E(B)^2
$$

##### **$a$** : Une constante qui a pour vocation d'ajuster le poids du rendement de la distribution des Lambda en fonction de **$n_{whale}$** et **$w_{whale}$**.

Soit :

$$a = 1 - \sum_{i=1}^{n_{whale}} w_{whale_i}$$

### Rendement du portefeuille

##### $$R_{portefeuille} = XW + Y$$
            
Où :
            
##### **$X$** : La distribution de rendement de nos *Lambdas* sélectionnés selon notre pouvoir de prédiction, soit **$N_{portefeuille} - n_{whale}$** valeurs.

La méthodologie est la suivante, nous récupérons
$N_{portefeuille} - n_{whale}$ valeurs que nous avons prédites
haussières. Nous avons :

-   Notre prédiction est correcte
$\frac{p_{h} \times p_{pc}}{p_{h} \times p_{pc} + \left( 1 - p_{h} \right) \times \left( 1 - p_{pc} \right)}$
et nous avons une hausse $r_{h}$

-   Notre prédiction est incorrecte
$\frac{\left( 1 - p_{h} \right) \times \left( 1 - p_{pc} \right)}{p_{h} \times p_{pc} + \left( 1 - p_{h} \right) \times \left( 1 - p_{pc} \right)}$
et nous avons une baisse $r_{b}$

Donc :

$$
E(X) = \left( \frac{p_{h} \times p_{pc}}{p_{h} \times p_{pc} + \left( 1 - p_{h} \right) \times \left( 1 - p_{pc} \right)} \right) \cdot r_{h} + \left( \frac{\left( 1 - p_{h} \right) \times \left( 1 - p_{pc} \right)}{p_{h} \times p_{pc} + \left( 1 - p_{h} \right) \times \left( 1 - p_{pc} \right)} \right) \cdot r_{b}
$$

Et sa variance :

$$
Var(X) = \frac{\left( \frac{p_{h} \times p_{pc}}{p_{h} \times p_{pc} + \left( 1 - p_{h} \right) \times \left( 1 - p_{pc} \right)} \right) \cdot r_{h}^{2} + \left( \frac{\left( 1 - p_{h} \right) \times \left( 1 - p_{pc} \right)}{p_{h} \times p_{pc} + \left( 1 - p_{h} \right) \times \left( 1 - p_{pc} \right)} \right) \cdot r_{b}^{2} - E(X)^{2}}{N_{portefeuille} - n_{whale}}
$$

##### **$Y$** : La distribution de rendement de nos *Whales* ajustés par la surpondération ou sous-pondération, soit **$n_{whale}$** valeurs avec un poids de **$w_{whale_i} + x_i$** ou **$w_{whale_i} - y_i$**.

L'espérance de rendement de cette distribution est la suivante :

-   Une probabilité $p_{h,i}^{whale}$ d'avoir un rendement $r_{h,i}^{whale}$ et de bien le prédire, donc
    $p_{h,i}^{whale} \times p_{pc} \times \left( w_{whale_i} + x_i \right) \times r_{h,i}^{whale}$

-   Une probabilité $p_{h,i}^{whale}$ d'avoir un rendement $r_{h,i}^{whale}$ et de mal le prédire, donc
    $p_{h,i}^{whale} \times \left( 1 - p_{pc} \right) \times \left( w_{whale_i} - y_i \right) \times r_{h,i}^{whale}$

-   Une probabilité $\left( 1 - p_{h,i}^{whale} \right)$ d'avoir un rendement $r_{b,i}^{whale}$ et de bien le prédire, donc
    $\left( 1 - p_{h,i}^{whale} \right) \times p_{pc} \times \left( w_{whale_i} - y_i \right) \times r_{b,i}^{whale}$

-   Une probabilité $\left( 1 - p_{h,i}^{whale} \right)$ d'avoir un rendement $r_{b,i}^{whale}$ et de mal le prédire, donc
    $\left( 1 - p_{h,i}^{whale} \right) \times \left( 1 - p_{pc} \right) \times \left( w_{whale_i} + x_i \right) \times r_{b,i}^{whale}$

Et sachant qu'on en a $n_{whale}$. Donc :

$$
E(Y) = \sum_{i=1}^{n_{whale}} \left\lbrack p_{h,i}^{whale} \times r_{h,i}^{whale} \times \left( p_{pc} \times \left( w_{whale_i} + x_i \right) + \left( 1 - p_{pc} \right) \times \left( w_{whale_i} - y_i \right) \right) + \left( 1 - p_{h,i}^{whale} \right) \times r_{b,i}^{whale} \times \left( p_{pc} \times \left( w_{whale_i} - y_i \right) + \left( 1 - p_{pc} \right) \times \left( w_{whale_i} + x_i \right) \right) \right\rbrack
$$

Et sa variance :

$$
Var(Y) = \sum_{i=1}^{n_{whale}} \left\lbrack p_{h,i}^{whale} \times (r_{h,i}^{whale})^2 \times \left( p_{pc} \times (w_{whale_i} + x_i)^2 + \left( 1 - p_{pc} \right) \times (w_{whale_i} - y_i)^2 \right) + \left( 1 - p_{h,i}^{whale} \right) \times (r_{b,i}^{whale})^2 \times \left( p_{pc} \times (w_{whale_i} - y_i)^2 + \left( 1 - p_{pc} \right) \times (w_{whale_i} + x_i)^2 \right) \right \rbrack - E(Y)^2
$$

##### **$W$** : La distribution de l'ajustement de la taille poche X, ajustée par les surpondérations ou sous-pondérations faites sur Y.

Comment on l'a vu précédemment, on a :

-   Une probabilité $p_{h,i}^{whale}$ d'avoir un rendement $r_{h,i}^{whale}$ et de bien le prédire, donc
    $p_{h,i}^{whale} \times p_{pc} \times \left( w_{whale_i} + x_i \right) \times r_{h,i}^{whale}$,
    donc un surajustement de $p_{h,i}^{whale} \times p_{pc} \times x_i$

-   Une probabilité $p_{h,i}^{whale}$ d'avoir un rendement $r_{h,i}^{whale}$ et de mal le prédire, donc
    $p_{h,i}^{whale} \times \left( 1 - p_{pc} \right) \times \left( w_{whale_i} - y_i \right) \times r_{h,i}^{whale}$,
    donc un surajustement de $p_{h,i}^{whale} \times \left( 1 - p_{pc} \right) \times y_i$

-   Une probabilité $\left( 1 - p_{h,i}^{whale} \right)$ d'avoir un rendement $r_{b,i}^{whale}$ et de bien le prédire, donc
    $\left( 1 - p_{h,i}^{whale} \right) \times p_{pc} \times \left( w_{whale_i} - y_i \right) \times r_{b,i}^{whale}$,
    donc un surajustement de $\left( 1 - p_{h,i}^{whale} \right) \times p_{pc} \times y_i$

-   Une probabilité $\left( 1 - p_{h,i}^{whale} \right)$ d'avoir un rendement $r_{b,i}^{whale}$ et de mal le prédire, donc
    $\left( 1 - p_{h,i}^{whale} \right) \times \left( 1 - p_{pc} \right) \times \left( w_{whale_i} + x_i \right) \times r_{b,i}^{whale}$,
    donc un surajustement de $\left( 1 - p_{h,i}^{whale} \right) \times \left( 1 - p_{pc} \right) \times x_i$

Donc :

$$
E(W) = 1 - \sum_{i=1}^{n_{whale}} \left( w_{whale_i} + \left( p_{h,i}^{whale} \times p_{pc} + \left( 1 - p_{h,i}^{whale} \right) \times \left( 1 - p_{pc} \right) \right) \times x_i - \left( p_{h,i}^{whale} \times \left( 1 - p_{pc} \right) + \left( 1 - p_{h,i}^{whale} \right) \times p_{pc} \right) \times y_i \right)
$$

Et sa variance :

$$
Var(W) = \sum_{i=1}^{n_{whale}} \left( w_{whale_i} + \left( p_{h,i}^{whale} \times p_{pc} + \left( 1 - p_{h,i}^{whale} \right) \times \left( 1 - p_{pc} \right) \right) \times x_i^2 - \left( p_{h,i}^{whale} \times \left( 1 - p_{pc} \right) + \left( 1 - p_{h,i}^{whale} \right) \times p_{pc} \right) \times y_i^2 \right) - E(W)^2
$$

Le détails des autres calculs est disponible dans la partie développée.
