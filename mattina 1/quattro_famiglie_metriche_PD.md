# Le quattro famiglie di metriche per un modello di PD

**Corso «Deep Learning per il Rischio di Credito: stima della PD con PyTorch»** — documento di riferimento.

Un modello di PD (*probability of default*, probabilità di inadempienza) emette, per ogni cliente, un numero tra 0 e 1. Quel numero può essere interrogato in quattro modi diversi, e a ogni modo corrisponde una famiglia di metriche: la **classificazione** giudica le decisioni prese col modello, l'**ordinamento** giudica la graduatoria che il modello produce, la **probabilità** giudica la verità dei valori stimati, il **tempo** giudica la tenuta di tutto questo quando il portafoglio cambia. Nessuna famiglia sostituisce le altre: rispondono a quattro domande diverse sullo stesso modello, e una valutazione completa le percorre tutte.

| Famiglia | Domanda a cui risponde | Metriche principali | Richiede una soglia? |
|---|---|---|---|
| 1. Classificazione | le decisioni sì/no prese col modello sono giuste? | accuracy, precision, recall, F1 | sì — esplicita o implicita |
| 2. Ordinamento (ranking) | la graduatoria dal meno al più rischioso è giusta? | ROC/AUC, Gini, PR-AUC, KS | no — viene scandita tutta |
| 3. Probabilità (calibrazione) | i valori stimati sono veri? | Brier, reliability diagram, test binomiale e di Jeffreys | no — non esistono classi |
| 4. Tempo (stabilità) | tutto questo regge sul portafoglio di oggi? | PSI, CSI, backtesting | no |

## Perimetro e convenzioni

- Contesto: classificazione binaria (default sì/no entro un orizzonte fisso, tipicamente 12 mesi) il cui output utile è la probabilità; portafogli sbilanciati, con tassi di default indicativamente tra l'1% e il 10%.
- Notazione: *PD stimata* è l'output del modello per un cliente; *esito* è ciò che poi accade (1 = default, 0 = no); *soglia s* è il punto di taglio che trasforma una probabilità in una classe.
- Tutte le soglie numeriche di lettura citate (Gini 40%/60%, PSI 0,10/0,25, p-value al 5%) sono **prassi di validazione**, non valori imposti da una norma.

---

## 1. Metriche di classificazione — giudicano la decisione

### Da dove nascono

Queste metriche non si calcolano sulle probabilità ma sulle **classi previste**, e una classe nasce solo applicando una soglia: classe = 1 se PD stimata ≥ s, altrimenti 0. La soglia può essere esplicita oppure **implicita**, cioè presente ma nascosta dentro lo strumento:

- `predict()` di scikit-learn applica silenziosamente s = 0,5 alle probabilità;
- un albero di decisione assegna alla foglia la classe di maggioranza: è una soglia al 50% sulla proporzione di default nella foglia;
- una SVM classifica col segno della distanza dal piano separatore: soglia a zero sul punteggio;
- un k-NN a voto di maggioranza mette la soglia sul conteggio dei voti.

La regola generale: **qualunque decisione binaria presa a partire da un'informazione continua implica un criterio di taglio**. L'unico modo di non avere una soglia è non decidere, cioè restituire il numero.

### La matrice di confusione

Fissata la soglia, ogni cliente finisce in una delle quattro celle della **matrice di confusione** — più una contabilità che una metrica:

| | previsto «buono» (0) | previsto «default» (1) |
|---|---|---|
| **reale buono (0)** | TN — vero negativo | FP — falso positivo (falso allarme) |
| **reale default (1)** | FN — falso negativo (default mancato) | TP — vero positivo |

La matrice rende visibile il fatto centrale della famiglia: gli errori sono **due**, di natura e costo diversi.

### Le metriche derivate

- **Accuracy** = (TP + TN) / totale: la quota di risposte giuste. È la metrica più intuitiva e la più ingannevole sui fenomeni sbilanciati: su un portafoglio col 3% di default, il «modello» che risponde sempre *nessun default* raggiunge il 97% di accuracy senza trovare un solo default.
- **Precision** = TP / (TP + FP): tra i segnalati come default, quanti lo sono davvero.
- **Recall** = TP / (TP + FN): tra i default veri, quanti ne troviamo. (Il gemello sul lato opposto, FPR = FP / (FP + TN), servirà alla curva ROC.)
- **F1** = media armonica di precision e recall = 2·P·R / (P + R). La media armonica — il reciproco della media dei reciproci — è dominata dal valore più basso: se il recall è 0, F1 è 0, qualunque sia la precision. Per questo sintetizza meglio della media ordinaria due grandezze in tensione: premia solo chi è bravo su entrambe.

### Uso corretto e limiti

Sono le metriche giuste quando il modello alimenta una **decisione operativa** con soglia dichiarata: un sistema di allerta, un filtro accetta/rifiuta. In quel caso la soglia si sceglie sui **costi dei due errori** (perdita attesa su un default accettato contro margine perso su un buon cliente rifiutato), che quasi mai sono simmetrici. Il limite strutturale: tutte queste metriche misurano la **coppia (modello, soglia)**, non il modello — cambiando s cambiano tutte, a probabilità identiche. E con sbilanciamento forte, a soglia 0,5 risultano quasi mute: quasi nessuna PD stimata supera il taglio.

---

## 2. Metriche di ordinamento (ranking) — giudicano la graduatoria

### La domanda e la proprietà chiave

Qui la domanda cambia natura: non più «le decisioni sono giuste?» ma **«il modello mette i clienti nell'ordine giusto, dal meno al più rischioso?»**. Le metriche di questa famiglia guardano solo le *posizioni relative* dei punteggi, non i loro valori: sono **invarianti per trasformazioni monotone crescenti** (raddoppiare tutte le probabilità, prenderne il logaritmo, ricalibrarle: la graduatoria non si muove, e queste metriche nemmeno) e **non richiedono di scegliere alcuna soglia** — la scandiscono tutta, o la ignorano.

### ROC, AUC, Gini

- La **curva ROC** traccia recall (TPR) contro tasso di falsi allarmi (FPR) *per tutte le soglie possibili*: è la fotografia completa dei compromessi disponibili; la diagonale è il caso (modello che tira a indovinare).
- L'**AUC** è l'area sotto la ROC (0,5 = caso, 1 = perfetto), con un'interpretazione elegante: è la probabilità che, estratti a caso un futuro default e un buon cliente, il primo riceva un punteggio più alto del secondo. Pura qualità di graduatoria.
- Il **Gini** = 2·AUC − 1 porta la stessa informazione nella scala in cui 0 = caso e 1 = perfezione: è il dialetto della validazione creditizia (coincide con il Somers' D e con l'Accuracy Ratio della curva CAP). Lettura di prassi: sotto il 40% debole, 40–60% accettabile, oltre il 60% buono.

### PR-AUC e KS

- La **PR-AUC** è l'area sotto la curva precision–recall al variare della soglia. Con classi rare è più severa e più informativa della ROC: il suo livello «caso» non è 0,5 ma il tasso di default stesso.
- Il **KS** (Kolmogorov–Smirnov) è la massima distanza tra le distribuzioni cumulate dei punteggi di buoni e cattivi (equivalente al massimo di TPR − FPR su tutte le soglie): la tradizione del credit scoring.

### Uso corretto e limiti

È il cuore della valutazione di una PD: la **discriminazione** (sinonimo di ordinamento in questo contesto). Il limite è speculare alla forza: queste metriche non dicono *nulla* sui valori. Un modello può ordinare in modo eccellente e stimare sistematicamente il doppio del rischio vero: Gini ottimo, numeri inservibili.

---

## 3. Metriche di probabilità (calibrazione) — giudicano i valori

### Perché i valori contano

Una PD non è (solo) un classificatore: è un numero che entra in **calcoli**. L'accantonamento atteso è circa PD × LGD × EAD (perdita in caso di default per esposizione al momento del default); il pricing aggiustato per il rischio e il requisito di capitale usano la PD come quantità; l'orizzonte lifetime di IFRS 9 la proietta negli anni. La proprietà richiesta si chiama **calibrazione**: tra tutti i clienti a cui diciamo «PD ≈ x%», la quota che va effettivamente in default deve essere ≈ x%. Un errore di livello produce conti sbagliati anche con una graduatoria perfetta.

### I tre strumenti

- **Brier score** = media di (PD stimata − esito)². Un voto complessivo, da leggere mai in assoluto ma contro il riferimento del «modello piatto» che assegna a tutti il tasso medio: il suo Brier vale tasso × (1 − tasso) — circa 0,074 con l'8% di default. Più basso = meglio.
- **Reliability diagram**: si raggruppano i clienti per fasce di PD stimata e si traccia, per ogni fascia, la PD media stimata contro il tasso di default osservato. La diagonale è la calibrazione perfetta; punti **sopra** la diagonale segnalano **sottostima** (il caso che allarma la Vigilanza), punti sotto segnalano sovrastima (prudente, ma costosa in pricing e accantonamenti).
- **Test per fascia di rating**: la lettura statistica, fascia per fascia. Il **test binomiale esatto** chiede: se la PD stimata fosse quella vera, osservare questi default su questi clienti sarebbe plausibile? Il **test di Jeffreys** — previsto dalle istruzioni BCE per il *validation reporting* dei modelli interni — è la variante bayesiana: con d default su n clienti, la distribuzione a posteriori della PD vera è una Beta(d + ½, n − d + ½), e il p-value è la probabilità che la PD vera sia ≤ di quella stimata. In entrambi, p-value piccolo (per prassi: sotto il 5%) = la fascia sottostima il rischio.

### Calibrare non è riaddestrare

La calibrazione si corregge con un **post-processing**: si adatta una funzione sulle probabilità del campione di validazione (Platt scaling, regressione isotonica, temperature scaling per le reti) e la si applica ai dati nuovi, senza toccare il modello. Essendo trasformazioni monotone, **non muovono il Gini**: ordinamento e calibrazione sono proprietà separate, si misurano separate e si correggono separate.

### Il caso low-default

Su portafogli con pochi default osservati (grandi imprese, banche, sovrani) i test di calibrazione hanno **bassa potenza**: non vedere uno scostamento non significa assolverlo. Lì la pratica si sposta su intervalli di confidenza (DeLong, bootstrap), stime prudenziali alla Pluto–Tasche e aggregazione di più anni o portafogli (pooling).

---

## 4. Metriche di stabilità nel tempo — giudicano la tenuta

### La domanda

Il modello è stato stimato sul portafoglio di ieri e lavora su quello di oggi: mix di clientela, politiche commerciali e congiuntura cambiano. La quarta famiglia chiede: **le prime tre risposte valgono ancora?**

### PSI e CSI

Il **PSI** (*Population Stability Index*) misura quanto la distribuzione attuale dei punteggi si è allontanata da quella di sviluppo. Si dividono i punteggi in fasce e si calcola PSI = Σ (p − q) · ln(p / q), dove p e q sono le quote di popolazione nella fascia, oggi e allo sviluppo. Lettura di prassi: sotto 0,10 stabile; 0,10–0,25 da tenere d'occhio; oltre 0,25 deriva significativa. Il **CSI** applica la stessa formula variabile per variabile, per capire *chi* si è spostato. In linguaggio ML, entrambi misurano il *data drift*.

### Ciò che il PSI non vede

Il PSI confronta distribuzioni, non relazioni: se la popolazione resta ferma ma cambia il **legame** tra variabili e default (*concept drift* — ad esempio dopo uno shock macroeconomico), il PSI resta sereno mentre il modello perde mordente. L'antidoto è il **backtesting periodico**: ricalcolare su finestre recenti le metriche delle famiglie 2 e 3 — Gini e calibrazione — e confrontarle con i valori di sviluppo.

### Gli esiti del monitoraggio

Quando qualcosa si muove, le leve sono tre, in ordine crescente di invasività: **ricalibrare** (se l'ordinamento tiene ma i livelli no), **riaddestrare** (se la discriminazione degrada), **sostituire col challenger** (se un modello alternativo, valutato sulle stesse quattro famiglie, fa stabilmente meglio). Ogni passaggio va documentato nella governance del modello.

---

## Le quattro famiglie insieme

Il filo che le lega: ogni famiglia copre il punto cieco della precedente. Le metriche di classificazione giudicano decisioni ma dipendono da una soglia; quelle di ordinamento eliminano la soglia ma ignorano i valori; quelle di probabilità verificano i valori ma fotografano un solo istante; quelle di tempo sorvegliano l'istante successivo. Una **scheda di validazione** completa riporta tutte e quattro le prospettive sullo stesso campione di test — e la Vigilanza se le aspetta tutte, perché un modello può essere eccellente in una famiglia e indifendibile in un'altra.

## Riferimenti essenziali

- Definizione di default: art. 178 CRR, con le soglie di rilevanza EBA; stima della PD: linee guida EBA/GL/2017/16.
- Test di Jeffreys per fascia: istruzioni BCE per il *validation reporting* dei modelli interni.
- Intervalli di confidenza per l'AUC: DeLong, E.R. et al. (1988), *Biometrics* 44(3); stime prudenziali low-default: Pluto, K., Tasche, D. (2005), «Thinking positively», *Risk*.
- Le soglie di lettura citate (Gini 40%/60%, PSI 0,10/0,25, p-value 5%) sono prassi di validazione, dichiarate come tali.
