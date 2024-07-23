## Sujet de recherche sur l'allocation tactique

Les entreprises de grande capitalisation, telles que les GAFAM (Google,
Apple, Facebook, Amazon, Microsoft) et Nvidia, jouent un rôle moteur
dans la performance de l\'indice S&P 500. En raison de leur poids
significatif, ces actifs peuvent influencer de manière disproportionnée
les rendements de l\'indice.

Cela pose la question de savoir si, dans le cadre d\'une gestion où nous
cherchons à battre un indice, nous sommes dépendants de ces valeurs.
Peut-on réellement s\'en séparer sans prendre des risques excessifs ?
Ignorer ces titres pourrait entraîner une sous-performance par rapport à
l\'indice, surtout s\'ils continuent de dominer les rendements.
Cependant, une trop grande dépendance envers ces valeurs peut également
accroître la volatilité et le risque de notre portefeuille.

Ainsi, la problématique que rencontre l'équipe Quantitative se concentre
sur la manière dont ils devraient se positionner sur les *Whales*, ces
entreprises dont la capitalisation boursière est si significative
qu\'elles dominent la performance de l\'indice. On gardera ce terme, qui
signifie \"baleine\" en anglais, et qui est l\'appellation courante que
nous utilisons et que j'ai employée dans mes différentes rédactions.

Pour tenter de répondre à ce sujet qui m'a été proposé, il m'a été
suggéré de commencer par répondre à un exercice théorique qui simplifie
notre univers d\'investissement et nous permet d\'analyser différentes
stratégies d\'allocation de portefeuille.

L'exercice est le suivant. Pour mieux comprendre l'impact des *Whales*,
nous supposons un univers d'investissement de 100 valeurs distinctes où
les probabilités de rendements haussiers ou de baissiers sont
symétriques, et où nous définissons trois indices représentatifs :

-   **I1 (Équipondéré)** : Chaque valeur a une pondération égale de 1%.

-   **I2 (Concentration Modérée)** : Cinq valeurs dominent, chacune
    pesant 10%, tandis que les 95 autres valeurs se partagent
    équitablement les 50% restants.

-   **I3 (Concentration Extrême)** : Une seule valeur domine le marché
    avec 50% du poids total. Les 99 autres valeurs se partagent les 50%
    restants.

L'équipe Quantitative dispose d'outils de prédiction, et donc, dans le
cadre de l'exercice, nous allons supposer que nous utilisons une
stratégie de détection de Momentum[^1] qui anticipe les tendances de
hausse ou de baisse avec une probabilité de succès de 60%. En réalité,
une telle probabilité de succès est très difficilement atteignable mais
pour pouvoir répondre simplement à ce premier exercice, il est important
de poser un cadre hypothétique. Cela implique que notre stratégie est
légèrement meilleure que le hasard (probabilité de succès de 50%) et
peut potentiellement générer de l'alpha, c'est-à-dire une surperformance
par rapport au marché.

Parmi les 100 valeurs de notre univers d'investissement, nous devons
constituer un portefeuille de 30 valeurs pour tenter de battre chacun
des trois indices définis précédemment.

Avec ces différentes informations, nous devons maintenant tenter de
répondre à deux questions :

-   En tenant compte de nos capacités de prédiction, quelle est la (ou
    les) constitution(s) de portefeuille que nous pensons optimale(s)
    pour battre l'indice I1, l'indice I2, et l'indice I3 sur le long
    terme ?

-   Existe-t-il une méthode générique efficace en fonction des
    allocations de l'indice de référence à battre ? Si oui, laquelle ?

Nous allons passer rapidement sur la première question en proposant une
réponse succincte. Nous tenterons d'apporter un résultat plus précis et
détaillé dans notre réponse à la seconde question.

### Réponse à la Question 1 : Constitution Optimale des Portefeuilles

Par souci de clarté, dans la suite du rapport, les valeurs équipondérées
(les valeurs autres que les grandes capitalisations) seront appelées les
valeurs *Lambdas*, et les valeurs avec une plus forte pondération seront
appelées *Whales.*

Pour chaque indice de référence (I1, I2, I3), nous devons examiner la
meilleure façon de répartir nos investissements sur les 30 valeurs de
notre portefeuille en utilisant notre méthode de détection de Momentum.
Voici nos premières intuitions :

#### Indice I1 (Équipondéré)

Puisque toutes les valeurs ont un poids égal dans l\'indice I1 et que
nous n\'avons que nos prédictions à disposition, notre objectif sera de
diversifier largement pour maximiser les opportunités de Momentum. Une
stratégie diversifiée consisterait à sélectionner 30 valeurs prédites
comme haussières et à les répartir de manière équipondérée. N\'ayant pas
de raison d\'attribuer plus de pondération à une valeur plutôt qu\'à une
autre, cette approche devrait permettre de surpasser l\'indice.

De plus, une répartition équipondérée de nos valeurs réduit la variance
du portefeuille en diversifiant le risque spécifique à chaque action.
Plus nous avons d\'actions dans notre portefeuille, plus le risque est
réduit. En combinant ces éléments, nous pouvons justifier la création
d\'un portefeuille de $N_{portefeuille}$ actions prédites haussières
comme suit : en utilisant notre pouvoir de prédiction $p_{pc}\%$ et la
probabilité connue de hausse $p_{h}$, nous sélectionnons les actions
ayant la plus grande probabilité de rendement positif pour maximiser
l\'espérance de rendement global du portefeuille, tout en gérant le
risque à travers l\'équipondération.

#### Indice I2 (Concentration Modérée)

Dans l\'indice I2, nous avons 5 valeurs représentant chacune 10% de
l'indice. Cette information supplémentaire, par rapport à une
répartition équipondérée, peut justifier de traiter ces valeurs
différemment. Nous pourrions proposer la même réponse que pour l'indice
I1, mais nous allons vite nous heurter à un problème. Si nos valeurs à
10% ont une tendance haussière et que nous avons bien anticipé cette
hausse, la sous-pondération de ces *Whales* entraînera une
sous-performance par rapport à l\'indice. En effet, les *Whales* ayant
une pondération importante, leur performance positive aura un impact
significatif sur l'indice, et notre portefeuille ne bénéficiera pas
pleinement de cette hausse si nous les sous-pondérons.

Pour aborder cette situation, nous proposons donc de surpondérer ou de
sous-pondérer les *Whales* en fonction de nos prédictions. Prenons un
exemple pour illustrer cette approche : parmi les 5 *Whales*, si 4 sont
prédites comme haussières, nous décidons de les surpondérer de $x$ %,
par exemple de 5%, les portant à 15% (10% + 5%) du portefeuille. La
cinquième, prédite baissière, est sous-pondérée de $y$ %, par exemple de
4%, réduisant son poids à 6%. Ainsi, la somme des poids des 5 valeurs
passe de 50% à 66% (4 x 15% + 6%). Les 25 autres valeurs, les *Lambdas*,
sont réparties de manière égale sur les 34% restants (100% - 66%).

#### Indice I3 (Concentration Extrême)

L'approche pour l'indice I3 serait similaire à celle de l'indice I2,
mais avec une seule valeur représentant 50% de l\'indice. La
surperformance de notre portefeuille dépendra fortement de cette
*Whale*. Si notre prédiction indique une hausse, cette valeur devrait
constituer une part significative de notre portefeuille. En revanche, si
elle est prédite baissière, nous devrions la sous-pondérer par rapport à
sa taille initiale dans l'indice.

Pour traiter cette situation, nous pouvons ajuster les pondérations
comme suit :

-   Si la *Whale* est prédite haussière, nous la surpondérons de $x$ %.

-   Si elle est prédite baissière, nous la sous-pondérons de $y$ %.

Le tout sera accompagné d\'une diversification parmi les 29 autres
valeurs *Lambdas* pour équilibrer le risque, de manière similaire aux 25
valeurs dans l'indice I2. En appliquant cette méthode de surpondération
et de sous-pondération, nous pouvons optimiser notre portefeuille pour
répondre aux variations des *Whales* tout en gérant le risque global.

Nous parlons de surpondération ou de sous-pondération de nos *Whales*,
mais de combien exactement ? Quelle méthode maximisera le rendement, et
quelle méthode minimisera le risque ? Est-il possible de trouver un
juste milieu ? Ce sont autant de questions auxquelles nous allons tenter
de répondre. Pour vérifier nos intuitions, nous abordons donc la seconde
question de manière théorique.

### Réponse à la Question 2 : Méthode Générique d\'Allocation

Nous allons aborder le sujet en supposant que l\'indice contient
$N_{indice}$ valeurs, et que notre portefeuille contient
$N_{portefeuille}$ valeurs. Notre univers présente une probabilité
$p_{h}^{whale}$ de rendements haussiers $r_{h}^{whale}$ et une probabilité $1 - p_{h}^{whale}$ 
de rendements baissiers $r_{b}^{whale}$ associés aux *Whales*. Et l'équivalent pour les *Lambdas*, $p_{h}^{lambda}$ de rendements haussiers $r_{h}^{lambda}$ et une probabilité $1 - p_{h}^{lambda}$ de rendements baissiers $r_{b}^{lambda}$. Nous disposons également d'un outil de
prédiction de Momentum avec une précision de $p_{pc}\%.$

Dans notre indice, nous pouvons donc avoir $n_{whale}$ valeurs, chacune
représentant une pondération de $w_{whale_i}$, le tout regroupé dans un vecteur tel que :

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

De la sorte, nous avons univers commun à tous les indices :

-   $N_{indice} = 100$

-   $N_{portefeuille} = 30$

-   $p_{pc} = 0.6$

-   $p_{h}^{lambda} = 0.5$

-   $r_{h}^{lambda} = 1$

-   $r_{b}^{lambda} = - 1$

Et pour nos indices, nous avons les paramètres suivants :

L'indice 1 :

-   $n_{whale} = 0$

-   $w_{whale} = 0$

L'indice 2 :

-   $n_{whale} = 5$

- $w_{whale} = \begin{pmatrix}
0.1 \\
0.1 \\
0.1 \\
0.1 \\
0.1 \\
\end{pmatrix}$

-   $p_{h}^{whale} = \begin{pmatrix}
0.5 \\
0.5 \\
0.5 \\
0.5 \\
0.5 \\
\end{pmatrix}$

-   $r_{h}^{whale} = \begin{pmatrix}
1 \\
1 \\
1 \\
1 \\
1 \\
\end{pmatrix}$

-   $r_{b}^{whale} = \begin{pmatrix}
-1 \\
-1 \\
-1 \\
-1 \\
-1 \\
\end{pmatrix}$

L'indice 3 :

-   $n_{whale} = 1$

- $w_{whale} = \begin{pmatrix}
0.5 \\
\end{pmatrix}$

-   $p_{h}^{whale} = \begin{pmatrix}
0.5 \\
\end{pmatrix}$

-   $r_{h}^{whale} = \begin{pmatrix}
1 \\
\end{pmatrix}$

-   $r_{b}^{whale} = \begin{pmatrix}
-1 \\
\end{pmatrix}$

Rappelons rapidement nos intuitions :

Pour l'indice I1, où toutes les valeurs ont le même poids
$(1/N_{indice})$, nous n'avons que notre pouvoir de prédiction
$(p_{pc})$ et la probabilité de hausse $(p_{h})$ pour nous guider. Nous
sélectionnons $(N_{portefeuille})$ actions avec la plus grande
probabilité de rendement positif et les répartissons de manière
équipondérée dans notre portefeuille. Cette diversification réduit la
variance en minimisant le risque spécifique à chaque action.

Pour les indices I2 et I3, où certaines valeurs *Whales* ont une grande
pondération $w_{whale}$, nous devons les traiter différemment. Ignorer
ces valeurs importantes $n_{whale}$ serait risqué. Si les *Whales* sont
majoritairement haussières, notre portefeuille risquerait de
sous-performer l'indice si nous ne les incluons pas.

Nous décidons donc d'inclure les Whales dans notre portefeuille. Nous surpondérons les Whales que nous prévoyons haussières de $x$ % et sous-pondérons celles que nous prévoyons baissières de $y$ %. Par exemple, si une Whale est baissière, nous pouvons imaginer la sous-pondérer de $y_i = w_{whale_i}$, ce qui équivaut à l'exclure complètement du portefeuille.

Les vecteurs $x$ et $y$ sont définis comme suit :

- $x$ : le vecteur de surpondération des rendements haussiers associés aux whales (vecteur de dimension $n_{whale}$).
- $y$ : le vecteur de sous-pondération des rendements baissiers associés aux whales (vecteur de dimension $n_{whale}$).

La formule de pondération devient alors :

$
w_{whale_i, haussière} = w_{whale_i} + x_i
$

$
w_{whale_i, baissière} = w_{whale_i} - y_i
$


Le reste du portefeuille $(N_{portefeuille} - n_{whale})$ est composé de
valeurs *Lambdas* avec des pondérations égales sur le reste de
l'exposition disponible. Cette exposition disponible est calculée comme
étant 100% moins la somme des nouveaux poids des *Whales ?*

Ainsi, si les nouveaux poids des *Whales* totalisent
$W_{whale,total}\%$, l'exposition disponible pour les valeurs *Lambdas*
est ${100\% - W}_{whale,total}\%$. Les valeurs *Lambdas* sont alors
réparties équitablement sur cette portion restante du portefeuille.

L'optimisation de notre performance passe par le réglage précis des
pondérations ($x$ et $y$) des *Whales* en fonction de nos prédictions.
Nous ajustons notre portefeuille pour inclure ou exclure les *Whales*
selon notre pouvoir de prédiction, afin de maximiser l'espérance de nos
rendements tout en gérant le risque, soit minimiser notre variance.

Il est possible de retrouver en annexe les rappels des fondamentaux sur
les calculs d'espérance et de variance. Chacune des équations a été
testée en parallèle de simulations pour vérifier si les résultats
théoriques sont bien en adéquation avec la pratique.

#### Espérance et Variance de l'indice

Nous allons commencer par définir le rendement de notre indice en posant
la distribution des rendements de l'indice tel que
$R_{indice} = La + B$, avec :

##### **$L$** : La distribution de rendement de nos *Lambdas*, soit **$N_{indice} - n_{whale}$** valeurs.

L'espérance de rendement de cette distribution est simple à calculer :

-   Nous avons une probabilité $p_{h}^{lambda}$ d'avoir un rendement $r_{h}^{lambda}$.

-   Une probabilité $1 - p_{h}^{lambda}$ d'avoir un rendement $r_{b}^{lambda}$.

Donc :

$$
E(L) = p_{h}^{lambda} \times r_{h}^{lambda} + \left( 1 - p_{h}^{lambda} \right) \times r_{b}^{lambda}
$$

Et sa variance :

$$
Var(L) = \frac{p_{h}^{lambda} \times r_{h}^{\lambda^{2}} + \left( 1 - p_{h}^{lambda} \right) \times r_{b}^{\lambda^{2}} - E(L)^{2}}{N_{indice} - n_{whale}}
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

Avec toutes ces informations, on peut donc maintenant déterminer
l'espérance des rendements de notre indice $E\left( R_{indice} \right)$
en fonction de nos différents paramètres.

$$E\left( R_{indice} \right) = E(L)a + E(B)$$

Et sachant qu'il n'y a pas de dépendance entre la distribution $L$ et
$B$, on peut définir la variance des rendements de l'indice
$Var\left( R_{indice} \right)$.

$$Var\left( R_{indice} \right) = Var(L)a^{2} + Var(B)$$

#### Espérance et Variance du portefeuille

Maintenant que nous avons défini les calculs d'espérance et de variance
pour notre indice, il est naturel de s'intéresser à l'équivalent pour
notre portefeuille. Nous allons définir le calcul de nos rendements de
portefeuille tel que $R_{portefeuille} = XW + Y$, avec :

##### **$X$** : La distribution de rendement de nos *Lambdas* sélectionnés selon notre pouvoir de prédiction, soit **$N_{portefeuille} - n_{whale}$** valeurs.

La méthodologie est la suivante, nous récupérons
$N_{portefeuille} - n_{whale}$ valeurs que nous avons prédites
haussières. Nous avons :

-   Notre prédiction est correcte
$\frac{p_{h}^{lambda} \times p_{pc}}{p_{h}^{lambda} \times p_{pc} + \left( 1 - p_{h}^{lambda} \right) \times \left( 1 - p_{pc} \right)}$
et nous avons une hausse $r_{h}^{lambda}$

-   Notre prédiction est incorrecte
$\frac{\left( 1 - p_{h}^{lambda} \right) \times \left( 1 - p_{pc} \right)}{p_{h}^{lambda} \times p_{pc} + \left( 1 - p_{h}^{lambda} \right) \times \left( 1 - p_{pc} \right)}$
et nous avons une baisse $r_{b}^{lambda}$

Donc :

$$
E(X) = \left( \frac{p_{h}^{lambda} \times p_{pc}}{p_{h}^{lambda} \times p_{pc} + \left( 1 - p_{h}^{lambda} \right) \times \left( 1 - p_{pc} \right)} \right) \cdot r_{h}^{lambda} + \left( \frac{\left( 1 - p_{h}^{lambda} \right) \times \left( 1 - p_{pc} \right)}{p_{h}^{lambda} \times p_{pc} + \left( 1 - p_{h}^{lambda} \right) \times \left( 1 - p_{pc} \right)} \right) \cdot r_{b}^{lambda}
$$

Et sa variance :

$$
Var(X) = \frac{\left( \frac{p_{h}^{lambda} \times p_{pc}}{p_{h}^{lambda} \times p_{pc} + \left( 1 - p_{h}^{lambda} \right) \times \left( 1 - p_{pc} \right)} \right) \cdot r_{h}^{\lambda^{2}} + \left( \frac{\left( 1 - p_{h}^{lambda} \right) \times \left( 1 - p_{pc} \right)}{p_{h}^{lambda} \times p_{pc} + \left( 1 - p_{h}^{lambda} \right) \times \left( 1 - p_{pc} \right)} \right) \cdot r_{b}^{\lambda^{2}} - E(X)^{2}}{N_{portefeuille} - n_{whale}}
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

Finalement, une fois les espérances et variances de nos différentes
distributions calculés, nous pouvons déterminer l'espérance des
rendements de notre portefeuille $E\left( R_{portefeuille} \right)$.

$$E\left( R_{portefeuille} \right) = E(XW + Y) = E(XW) + E(Y) = E(X) \times E(W) + E(Y)$$

La difficulté réside plus dans le calcul de la variance des rendements
du portefeuille :

$$Var\left( R_{portefeuille} \right) = Var(XW + Y)$$

Effectivement, il y a une interdépendance entre la distribution $W$ et
$Y$, puisque l'ajustement dans la distribution $W$ dépend directement
des valeurs et donc des poids attribués dans la distribution $Y$. Donc :

$$Var(XW + Y) = Var(XW) + Var(Y) + 2Cov(XW,Y)$$

Or X et W sont indépendants, alors :

$$Var(XW) = E(X)^{2} \times Var(W) + E(W)^{2} \times Var(X) + Var(X) + Var(W)$$

Donc :

$$Var(XW + Y) = E(X)^{2} \times Var(W) + E(W)^{2} \times Var(X) + Var(X) + Var(W) + Var(Y) + 2Cov(XW,Y)$$

Il nous reste donc à définir la covariance :

$$Cov(XW,Y) = E\left( (XW) \cdot Y \right) - E(XW)E(Y)$$

On rappelle l'indépendance de X et W, donc $E(X \cdot W) = E(X)E(W)$ :

$$Cov(XW,Y) = E(XWY) - E(X)E(W)E(Y)$$

Comme expliqué dans le contexte, on suppose X indépendant de W et de Y,
et W et Y dépendants. Alors :

$$E(XWY) = E\left( X \cdot (WY) \right) = E(X)E(WY)$$

On a pu donc définir la covariance tel que :

$$Cov(XW,Y) = E(X)E(WY) - E(X)E(W)E(Y)$$

Voici la formule finale de la variance :

$$Var(XW + Y) =$$

$$E(X)^{2} \times Var(W) + E(W)^{2} \times Var(X) + Var(X) + Var(W) + Var(Y) + 2\left\lbrack E(X)E(WY) - E(X)E(W)E(Y) \right\rbrack$$

La seule inconnue dans cette équation est $E(WY)$, il nous reste donc à
la définir. N'ayant pas réussi à généraliser cette équation, nous
faisons une proposition de solution qui permet de développer
l'arbre de probabilité qui permet de déterminer $E(WY)$ en fonction de
$n_{whale}$. 

##### Calcul de l'espérance de $WY$

Définissons les matrices :

$
\text{values\_matrix} =
\begin{pmatrix}
r_{h,i}^{whale} \times (w_{whale_i} + x_i) \\
r_{h,i}^{whale} \times (w_{whale_i} - y_i) \\
r_{b,i}^{whale} \times (w_{whale_i} - y_i) \\
r_{b,i}^{whale} \times (w_{whale_i} + x_i)
\end{pmatrix}
$

$
\text{prob\_matrix} =
\begin{pmatrix}
p_{h,i}^{whale} \times p_{pc} \\
p_{h,i}^{whale} \times (1 - p_{pc}) \\
(1 - p_{h,i}^{whale}) \times p_{pc} \\
(1 - p_{h,i}^{whale}) \times (1 - p_{pc})
\end{pmatrix}
$

$
\text{adjust\_matrix} =
\begin{pmatrix}
w_{whale_i} + x_i \\
w_{whale_i} - y_i \\
w_{whale_i} - y_i \\
w_{whale_i} + x_i
\end{pmatrix}
$

Nous considérons toutes les combinaisons possibles des éléments des matrices ci-dessus sur la dimension $n_{whale}$.

Le produit des probabilités pour chaque combinaison est :

$
\prod_{i=1}^{n_{whale}} \text{prob\_matrix}_{\text{comb},i}
$

La somme des valeurs pour chaque combinaison est :

$
\sum_{i=1}^{n_{whale}} \text{value\_matrix}_{\text{comb},i}
$

La somme des ajustements pour chaque combinaison est :

$
\sum_{i=1}^{n_{whale}} \text{adjust\_matrix}_{\text{comb},i}
$

Le terme pour chaque combinaison est :

$
\text{term}_{\text{comb}} = \prod_{i=1}^{n_{whale}} \text{prob\_matrix}_{\text{comb},i} \times \left( \sum_{i=1}^{n_{whale}} \text{value\_matrix}_{\text{comb},i} \right) \times \left( 1 - \sum_{i=1}^{n_{whale}} \text{adjust\_matrix}_{\text{comb},i} \right)
$

L'espérance de $WY$ est la somme de tous ces termes :

$
E(WY) = \sum_{\text{comb}} \text{term}_{\text{comb}}
$

Où :
- $\text{value\_matrix}_{\text{comb},i}$ est $rhx\_matrix_i, rhy\_matrix_i, rby\_matrix_i, rbx\_matrix_i$ en fonction de la combinaison.
- $\text{prob\_matrix}_{\text{comb},i}$ est $prob\_matrix_{rhx,i}, prob\_matrix_{rhy,i}, prob\_matrix_{rby,i}, prob\_matrix_{rbx,i}$ en fonction de la combinaison.
- $\text{adjust\_matrix}_{\text{comb},i}$ est $adjust\_matrix_{rhx,i}, adjust\_matrix_{rhy,i}, adjust\_matrix_{rby,i}, adjust\_matrix_{rbx,i}$ en fonction de la combinaison.

###### Calcul du nombre de combinaisons possibles

Soit $S = \{rhx, rhy, rbx, rby\}$, avec :

- $rhx$ : Rendement haussier correctement prédit
- $rhy$ : Rendement haussier incorrectement prédit
- $rbx$ : Rendement baissier incorrectement prédit
- $rby$ : Rendement baissier correctement prédit

 Nous voulons générer toutes les $4^n$ combinaisons possibles de $S$ de longueur $n$.

Les combinaisons peuvent être notées comme l'ensemble de tous les tuples de longueur $n$ formés à partir de $S$ :

$
\text{Combinations}(S, n) = \{(s_1, s_2, \ldots, s_n) \mid s_i \in S \text{ pour } i = 1, 2, \ldots, n\}
$

En notation mathématique :

$
\text{Combinations}(S, n) = \left\{ (s_1, s_2, \ldots, s_n) \mid s_i \in \{0, 1, 2, 3\} \right\}
$

Le nombre de combinaisons possibles lorsque nous générons toutes les combinaisons de $n$ éléments pris parmi un ensemble $S$ de 4 éléments est donné par :

$
|S|^n
$

Où :
- $|S|$ est la cardinalité de l'ensemble $S$ (c'est-à-dire le nombre d'éléments dans $S$).
- $n$ est le nombre de positions à remplir avec les éléments de $S$.

Pour $S = \{0, 1, 2, 3\}$, nous avons $|S| = 4$. Le nombre total de combinaisons possibles pour $n$ éléments est donc :

$
4^n
$

#### Espérance et Variance du delta entre le portefeuille et l'indice

Maintenant que nous connaissons le détail de l'espérance des rendements
de notre portefeuille, et notre indice, il est utile de rappeler que
nous cherchons à analyser la performance du portefeuille en relatif à la
performance de l'indice. Nous cherchons donc à définir l'espérance et la
variance du delta entre le rendement du portefeuille et celui de
l'indice.

Pour l'espérance, la marche à suivre est assez simple, d'après la
linéarité de l'espérance :

$$E(\Delta R) = E\left( R_{portefeuille} - R_{indice} \right)$$

$$E(\Delta R) = E\left( R_{portefeuille} \right) - E\left( R_{indice} \right)$$

Or $E\left( R_{portefeuille} \right)$ et $E\left( R_{indice} \right)$
sont déjà des variables que nous avons déterminées précédemment.

Pour la variance, la tâche est encore une fois un peu plus complexe.
Nous pouvons déterminer la variance de la sorte :

$$Var(\Delta R) = Var\left( R_{portefeuille} - R_{indice} \right)$$

$$Var(\Delta R) = Var\left( R_{portefeuille} \right) + Var\left( R_{indice} \right) - 2 \cdot Cov\left( R_{portefeuille},R_{indice} \right)$$

Nous allons nous heurter au même problème que pour déterminer
$Var\left( R_{portefeuille} \right)$ précédemment. Nous allons donc
détailler notre calcul :

$$Var(\Delta R) = Var(XW + Y) + Var\left( La + B \right) - 2 \cdot Cov\left( XW + Y,La + B \right)$$

Notre inconnu ici est la covariance entre le portefeuille et l'indice :

$$Cov(XW + Y,aL + B) = E\left\lbrack (XW + Y) \cdot (aL + B) \right\rbrack - E(XW + Y) \times E\left\lbrack (aL + B) \right\rbrack$$

La variable à définir est la suivante :
$E\left\lbrack (XW + Y) \cdot (aL + B) \right\rbrack$. Nous allons la
développer :

$$E\left\lbrack (XW + Y) \cdot (aL + B) \right\rbrack = E(XWaL + XWB + YaL + YB)$$

$$= E(XWaL) + E(XWB) + E(YaL) + E(YB)$$

$$= a \cdot E(XWL) + E(XWB) + a \cdot E(YL) + E(YB)$$

Pour continuer, nous allons nous confronter à la question de la
dépendance de nos distributions.

Après des tests de corrélation sur des simulations, on va faire
l'hypothèse suivante sur nos distributions :

-   X et aL sont dépendantes.

-   Y, B et W sont dépendantes.

La majeure partie de la covariance vient de la dépendance entre Y et B.
Maintenant que nous avons nos hypothèses de relation de dépendance, nous
pouvons continuer de développer
$E\left\lbrack (XW + Y) \cdot (aL + B) \right\rbrack$ :

$$E\left\lbrack (XW + Y) \cdot (aL + B) \right\rbrack = a \cdot E(XWL) + E(XWB) + a \cdot E(YL) + E(YB)$$

> $$= a \cdot E(W) \cdot E(XL) + E(X) \cdot E(WB) + a \cdot E(Y) \cdot E(L) + E(YB)$$

Les trois variables à définir ici sont $E(XL)$, $E(WB)$ et $E(YB)$. Pour
$E(WB)$ et $E(YB)$ nous pouvons réutiliser le procédé utilisé pour
$E(WY)$ que nous avons détaillé en annexe.

Pour le moment $E(XL)$ semble trop compliqué à définir. Nous décidons de
le simuler avec différents paramètres, et de lisser nos résultats
bruités par la simulation via un filtre de Savitzky-Golay (Exemple Cf.
Annexes).

#### Résultats et visualisation

Maintenant que nous avons :

-   $E\left( R_{indice} \right)$

-   $E\left( R_{portefeuille} \right)$

-   $E(\Delta) = E\left( R_{portefeuille} - R_{indice} \right)$

-   $Var\left( R_{indice} \right)$

-   $Var\left( R_{portefeuille} \right)$

-   $Var(\Delta) = Var\left( R_{portefeuille} - R_{indice} \right)$

Nous allons pouvoir étudier les solutions optimales pour nos différents
indices, que ce soit en maximisant nos espérances, en minimisant nos
variances, ou en trouvant le ratio optimal entre les deux qui
correspondrait à un Ratio de Sharpe ou un Ratio d'Information.

Le Ratio de Sharpe est un indicateur de performance qui mesure la
rentabilité excédentaire par unité de risque d'un portefeuille par
rapport à un indice de référence. Il est calculé en prenant la
différence entre le rendement attendu du portefeuille
$E\left( R_{portefeuille} \right)$ et le rendement d'un indice de
référence $E\left( R_{indice} \right)$, puis en divisant cette
différence par la volatilité du portefeuille
$Var\left( R_{portefeuille} \right)$. Plus le Ratio de Sharpe est élevé,
plus le rendement ajusté au risque est élevé. Ce ratio est
particulièrement utile pour comparer des portefeuilles ayant des niveaux
de risque différents, car il permet d'identifier celui qui offre le
meilleur rendement par unité de risque.

Si on réfléchit alors en termes de rendement absolue, c'est à dire en
maximisant nos rendements de portefeuille sans relatif à notre indice,
notre Ratio serait donc :

$Sharpe\ Ratio = \frac{E\left( R_{portefeuille} \right) - E\left( R_{indice} \right)}{\sigma\left( R_{portefeuille} \right)}$.

D'un autre côté, le Ratio d'Information mesure la performance d'un
portefeuille par rapport à un indice de référence, en tenant compte de
l'excès de rendement par rapport à cet indice
($E\left( R_{portefeuille} \right)$ - $E\left( R_{indice} \right)$) et
en le divisant par l'erreur de suivi (Tracking Error), qui représente la
volatilité de cette différence de rendement. Ce ratio est essentiel dans
une gestion benchmarkée, car il permet de déterminer dans quelle mesure
le gestionnaire de portefeuille a réussi à générer un excès de rendement
par rapport à l'indice tout en contrôlant le risque de déviation par
rapport à celui-ci.

Donc, quand notre réflexion se fait en relatif à un indice, on va
s'intéresser au Ratio d'Information :

$$Information\ Ratio = \frac{E\left( R_{portefeuille} - R_{indice} \right)}{Tracking\ Error}$$

Avec :

$$Tracking\ Error = \sqrt{Var\left( R_{portefeuille} - R_{indice} \right)} = \sigma\left( R_{portefeuille} - R_{indice} \right)$$

Soit :

$$Information\ Ratio = \frac{E\left( R_{portefeuille} - R_{indice} \right)}{\sigma\left( R_{portefeuille} - R_{indice} \right)}$$

# Annexes

## Fondamentaux du Calcul des Espérances

Le calcul des espérances est un concept clé en statistique et en
probabilité, offrant une mesure de la tendance centrale d'une
distribution de probabilité. Cela représente la valeur moyenne que vous
pouvez vous attendre à obtenir d'une variable aléatoire après un grand
nombre d'essais. Voici quelques-unes des propriétés fondamentales des
espérances :

### Définition de l'Espérance

Pour une variable aléatoire discrète $X$, prenant des valeurs $x_{i}$
avec les probabilités correspondantes $p_{i}$, l'espérance est donnée
par :

$$E(X) = \sum_{i}^{}x_{i}p_{i}$$

Où la somme porte sur toutes les valeurs possibles de $X$.

### Propriétés Fondamentales de l'Espérance

-   Espérance d'une Somme $E(X + Y) = E(X) + E(Y)$

-   Espérance d'un Produit pour des Variables Indépendantes
    $E(XY) = E(X) \cdot E(Y)\quad\text{si X et Y sont indépendantes}$

-   Espérance d'une Constante Multiplicative $E(aX) = a \cdot E(X)$

-   Linéarité de l'Espérance $E(aX + bY) = aE(X) + bE(Y)$

-   Espérance d'une Constante $E(a) = a$

## Fondamentaux du Calcul de Variance

La variance est une mesure statistique qui décrit la dispersion des
valeurs d'une variable aléatoire par rapport à sa moyenne. En d'autres
termes, elle mesure à quel point les valeurs d'un ensemble de données
sont éloignées les unes des autres. Voici quelques principes de base et
propriétés de la variance qui sont essentiels en statistique et en
probabilité.

### Définition de la Variance

La variance d'une variable aléatoire $X$, notée $Var(X)$ ou
$sigma_{X}^{2}$, est définie comme l'espérance du carré de l'écart entre
$X$ et son espérance $E(X)$ :

$$Var(X) = E\left\lbrack \left( X - E(X) \right)^{2} \right\rbrack$$

$$Var(X) = E\left( X^{2} \right) - E(X)^{2}$$

### Propriétés Fondamentales de la Variance

-   Variance d'une Constante : $Var(c) = 0$

-   Variance d'une Variable avec Constante Ajoutée :
    $Var(X + c) = Var(X)$

-   Variance d'une Variable Multipliée par une Constante :
    $Var(aX) = a^{2} \cdot Var(X)$

-   Variance de la Somme de Variables Indépendantes :
    $Var(X + Y) = Var(X) + Var(Y)\quad\text{si X et Y sont indépendantes}$

-   Variance de la Somme de Variables Dépendantes :
    $Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)\quad\text{si X et Y sont dépendantes}$

-   Variance de la Multiplication de Variables Indépendantes :
    $Var(X \cdot Y) = E(X)^{2} \times Var(Y) + E(Y)^{2} \times Var(X) + Var(X) + Var(Y)\quad\text{si X et Y sont indépendantes}$

-   Variance de la Multiplication de Variables Dépendantes :
    $Var(X \cdot Y) = E(X)^{2} \times Var(Y) + E(Y)^{2} \times Var(X) + Var(X) + Var(Y) + 2 \times E(X) \times E(Y) \times Cov(X,Y)$
    $\text{si X et Y sont dépendantes}$

## Fondamentaux du Calcul de Covariance

La covariance est une mesure statistique qui évalue le degré auquel deux
variables aléatoires varient ensemble. En d'autres termes, elle indique
si une augmentation de l'une des variables est associée à une
augmentation de l'autre variable (covariance positive), ou si une
augmentation de l'une est associée à une diminution de l'autre
(covariance négative). Voici quelques concepts de base et propriétés
importantes de la covariance.

### Définition de la Covariance

La covariance entre deux variables aléatoires $X$ et $Y$, notée
$Cov(X,Y)$, est définie comme :

$$Cov(X,Y) = E\left\lbrack \left( X - E(X) \right)\left( Y - E(Y) \right) \right\rbrack$$

$$Cov(X,Y) = E(XY) - E(X)E(Y)$$

Où $E(X)$ et $E(Y)$ sont les espérances (moyennes) de $X$ et $Y$,
respectivement.

### Propriétés Fondamentales de la Covariance

-   Covariance et Indépendance :
    $Cov(X,Y) = 0\quad\text{si X et Y sont indépendantes}$

-   Covariance d'une Variable avec Elle-même $Cov(X,X) = Var(X)$

-   Covariance et Constantes : $Cov(X,c) = 0$

-   Propriété de Symétrie : $Cov(X,Y) = Cov(Y,X)$

-   Linéarité de la Covariance :
    $Cov(aX + b,cY + d) = a \cdot c \cdot Cov(X,Y)$

-   Covariance de la Somme de Variables :
    $Cov\left( X_{1} + X_{2},Y \right) = Cov\left( X_{1},Y \right) + Cov\left( X_{2},Y \right)$

