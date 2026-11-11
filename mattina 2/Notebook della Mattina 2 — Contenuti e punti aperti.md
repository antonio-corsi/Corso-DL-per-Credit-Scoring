# Notebook della Mattina 2 — Contenuti e punti aperti

Nov 10, 2026 · @Antonio

## Assunzioni

Verifica pre-generazione del notebook della mattina 2: i contenuti sono chiari, restano tre punti da sciogliere prima di partire.

- Fonte dei contenuti: la sezione «Mattina 2 — Reti neurali su dati tabellari, fatte bene» del [documento di programma](https://claude.ai/artifact/X5xU1qvSz9WaxFBZ63FETE) (rev. 17).
- Decisioni della mattina 1 estese anche a questa: **un unico notebook per VS Code**, molto commentato in italiano, con assunzioni e glossario in testa, esercizi TODO in coda e stesso schema di collaudo.

## Le tre sezioni del notebook

Il notebook copre le tre coppie pillola/laboratorio del programma, in tre sezioni.

1. **Preprocessing per reti** — perché lo scaling (riportare le variabili su scale comparabili) è necessario per le reti e irrilevante per gli alberi; gestione dei valori mancanti (gli alberi li digeriscono, le reti no); encoding delle categoriche e soprattutto **embedding per l'alta cardinalità** al posto del one-hot; pipeline completa con `Dataset`/`DataLoader` e MLP con layer di embedding.
2. **Regolarizzazione e sbilanciamento** — dropout, batch normalization, early stopping, weight decay, con esperimenti controllati *una tecnica alla volta* letti sulle curve di apprendimento; pesi di classe vs **focal loss**, da implementare a mano in poche righe commentate (per il caso binario non è nel core di PyTorch, ed è un buon esercizio di lettura).
3. **Calibrazione e verdetto** — Platt scaling, regressione isotonica e temperature scaling sull'output della rete; tabella finale **champion/challenger** MLP vs XGBoost su Gini, PR-AUC, Brier e test di calibrazione; chiusura sulla discussione low-default.

## Esercizi opzionali e vincoli tecnici

- Esercizi opzionali già previsti dal programma: **embedding a dimensione variabile**; **focal loss vs pesi di classe** a parità di budget di training.
- Vincoli invariati rispetto alla mattina 1: la rete dell'ambiente di generazione è chiusa, quindi le celle PyTorch vengono verificate in sintassi e collaudate con mock; la verifica piena avviene al primo run nell'ambiente conda.

## Le tre domande prima di partire

1. **Dataset — il punto principale.** Il programma prevede un **campione di Home Credit Default Risk**, scelta azzeccata per questa mattina: categoriche ad alta cardinalità vere (tipo di organizzazione ≈58 livelli, occupazione ≈18) e valori mancanti realistici — esattamente ciò che serve a embedding e preprocessing. Kaggle però non è raggiungibile dall'ambiente di generazione (e richiede comunque un account). Proposta, sullo schema della mattina 1: **generatore sintetico locale «tipo Home Credit»** (≈60.000 domande di prestito, ≈8% di default, importi/età/anzianità/punteggi esterni, due categoriche ad alta cardinalità con effetto rischio iniettato, missing realistici) + **loader a doppio percorso** che usa il file reale `dati/home_credit_campione.csv` se presente; un eventuale `application_train.csv` scaricato da Kaggle viene campionato dal loader stesso. **Da confermare.**
2. **Champion da ri-addestrare.** Cambiando dataset rispetto al Taiwan, XGBoost va ri-addestrato in apertura del notebook su Home Credit (breve sezione «il champion sul nuovo terreno»), perché il confronto finale sia a parità di dati. È la lettura naturale del programma: segnalata per trasparenza.
3. **Random Forest — decisione ancora pendente.** Le opzioni: (a) baseline ufficiale accanto a logistica e XGBoost; (b) esercizio opzionale — *consigliata*; (c) nulla. Con la (b), l'esercizio può entrare già in questo notebook come TODO finale: «aggiungere Random Forest alla tabella champion/challenger».

## Nota a margine (non bloccante)

Il documento di programma recita ancora «12 notebook di laboratorio, tre per mattina»: da allineare alla decisione del notebook unico — 4 notebook di mattina + Notebook 0 + appendice GPU — al prossimo ritocco del documento.
