# Deep Learning per il Rischio di Credito — Programma di dettaglio

Oct 31, 2026 · @Antonio

## Scheda sintetica del corso

Il corso porta 15 analisti quantitativi di una banca internazionale dal machine learning classico (XGBoost, Random Forest, alberi decisionali) al deep learning con PyTorch per la stima della probabilità di default (PD), con interpretabilità e conformità alle aspettative della Vigilanza come filo conduttore.

| Voce | Dettaglio |
| --- | --- |
| Titolo | Deep Learning per il Rischio di Credito: stima della PD con PyTorch |
| Durata e formato | 4 mattine online, ore 9:00–13:00 con intervallo di 15 minuti a metà mattina (225 minuti netti), erogate in 2 edizioni identiche; rapporto teoria–pratica circa 30–70 |
| Destinatari | 15 analisti quantitativi per edizione (2 edizioni identiche, 30 persone in totale), con laurea STEM, dalle strutture Retail (big data), Large Corporate e SME |
| Prerequisiti | Python di base, scikit-learn, pratica consolidata con XGBoost, Random Forest e alberi decisionali |
| Tecnologie | Python, scikit-learn, XGBoost (come benchmark), PyTorch, SHAP, Captum, Fairlearn |
| Ambiente | Visual Studio Code con ambienti conda e accesso internet; laboratori dimensionati per CPU; GPU trattata in appendice dedicata (anche su Google Colab) |
| Lingua | Italiano, con il termine inglese tra parentesi alla prima occorrenza di ogni concetto tecnico |

Al termine del corso i partecipanti sapranno:

- valutare un modello di PD con le metriche attese dalla Vigilanza — discriminazione, calibrazione, stabilità — superando la sola accuracy;
- progettare, addestrare e regolarizzare reti neurali su dati tabellari di credito con PyTorch, confrontandole sistematicamente con XGBoost secondo l'approccio champion/challenger (modello in produzione contro modello sfidante);
- costruire modelli di PD a partire dalle sequenze grezze di transazioni di conto corrente e carte, il territorio in cui il deep learning aggiunge valore misurabile;
- spiegare le reti con Captum (DeepSHAP, Integrated Gradients) in continuità con lo SHAP già in uso, e misurare il bias con Fairlearn;
- impostare la documentazione del modello secondo le aspettative della Vigilanza, inclusa la gestione delle richieste di fine-tuning.

## Ipotesi di lavoro da validare

Su alcuni punti il cliente non ha fornito indicazioni: le ipotesi sotto adottano la prassi europea corrente e vanno validate dalla funzione di validazione interna prima dell'avvio; la loro sostituzione è un intervento minimo sui materiali.

| Punto | Ipotesi adottata |
| --- | --- |
| Definizione di default | Quella regolamentare europea dell'art. 178 CRR (Capital Requirements Regulation): 90 giorni di arretrato rilevante (past due) oppure improbabile adempimento (unlikely to pay, UTP), con le soglie di rilevanza EBA: 100 € per il retail, 500 € per il non retail, più l'1% relativo dell'esposizione |
| Orizzonte della PD | 12 mesi, target binario "default entro 12 mesi dalla data di osservazione", lo standard dei modelli IRB (Internal Ratings-Based, i modelli interni autorizzati per il calcolo del capitale); cenno in teoria alla PD lifetime richiesta da IFRS 9 |
| Soglia di discriminazione | Gini (= 2·AUC − 1) con lo schema a semaforo di prassi: sotto 40% debole, 40–60% accettabile, sopra 60% buono — dichiarando in aula che non sono valori imposti da una norma |
| Stabilità | PSI (Population Stability Index, indice dello spostamento della popolazione corrente rispetto a quella di sviluppo) con soglie 0,10 (stabile) e 0,25 (deriva significativa) |
| Calibrazione | Test binomiale e test di Jeffreys per fascia di rating — quest'ultimo previsto dalle istruzioni BCE per il validation reporting dei modelli interni — più Hosmer–Lemeshow come alternativa classica |
| Quadro normativo citato | CRR, linee guida EBA sulla stima della PD (EBA/GL/2017/16), istruzioni BCE sul validation reporting |
| Dati | Nessun dato reale della banca in aula: dataset pubblici e sintetici che replicano la struttura di transazioni di conto corrente e carte, etichettati secondo la definizione di default sopra |

## Struttura ricorrente di ogni mattina

Ogni mattina — dalle 9:00 alle 13:00, con un intervallo di 15 minuti a metà — segue lo stesso schema, senza minutaggi prefissati: tre pillole teoriche brevi, ciascuna seguita dal proprio laboratorio guidato. Il rapporto teoria–pratica di circa 30–70 si realizza sull'arco dell'intera mattina.

1. Apertura: obiettivi della mattina e richiamo della precedente
2. Pillola teorica 1 → Laboratorio 1 (guidato, con esercizi finali)
3. Pillola teorica 2 → Laboratorio 2
4. Intervallo di 15 minuti
5. Pillola teorica 3 → Laboratorio 3
6. Sintesi e anticipazione della mattina successiva — eventuali domande: al massimo 5 minuti

Ogni laboratorio si chiude con esercizi a difficoltà crescente; gli ultimi, opzionali e più avanzati, tengono impegnati i partecipanti con Python medio-alto senza rallentare il gruppo. I tempi dei singoli blocchi restano liberi, da adattare all'andamento dell'aula.

## Mattina 1 — Oltre l'accuracy: metriche per la PD e primi passi con PyTorch

Obiettivo: sostituire l'accuracy con il set di metriche che la Vigilanza si aspetta — è il fondamento su cui poggiano le tre mattine successive — e scrivere il primo training loop PyTorch.

| Blocco | Contenuto |
| --- | --- |
| Pillola 1 | Il problema PD: definizione di default (art. 178 CRR), orizzonte a 12 mesi, sbilanciamento delle classi (class imbalance). Perché l'accuracy inganna: con il 3% di default, il modello che risponde sempre "nessun default" raggiunge il 97% di accuracy restando inutilizzabile |
| Laboratorio 1 | Matrice di confusione, precision, recall, F1; curva ROC, AUC e Gini; PR-AUC e statistica KS (Kolmogorov–Smirnov). Confronto fra modello banale, regressione logistica e XGBoost: nasce la baseline champion che accompagnerà tutto il corso |
| Pillola 2 | La PD è una probabilità, non una classe: alimenta pricing, accantonamenti e capitale. Calibrazione: curve di calibrazione (reliability diagram), Brier score, test binomiale e test di Jeffreys per fascia di rating |
| Laboratorio 2 | Costruzione delle fasce di rating sul modello XGBoost; curve di calibrazione e test statistici per fascia; lettura dei risultati come li leggerebbe un validatore |
| Pillola 3 | PyTorch essenziale: tensori, autograd (il calcolo automatico dei gradienti), nn.Module, il training loop esplicito; parallelo con il fit/predict di scikit-learn per chi viene da lì |
| Laboratorio 3 | Primo MLP (multi-layer perceptron, percettrone multistrato) sul dataset tabellare; valutazione con tutte le metriche della mattinata e primo confronto a tre: logistica, XGBoost, rete |

Dataset: Taiwan Credit Card Default (UCI), fornito come file pronto. Esercizi opzionali avanzati: implementare la statistica KS da zero; variare l'architettura del MLP misurando l'effetto sul Gini.

## Mattina 2 — Reti neurali su dati tabellari, fatte bene

Obiettivo: trasformare il MLP della mattina 1 in un modello competitivo sui dati tabellari di credito, chiudendo con il confronto champion/challenger documentato contro XGBoost.

| Blocco | Contenuto |
| --- | --- |
| Pillola 1 | Preprocessing per reti: perché lo scaling (riportare le variabili su scale comparabili) è necessario per le reti e irrilevante per gli alberi — differenza non ovvia per chi viene da XGBoost; encoding delle categoriche; embedding (rappresentazioni numeriche dense apprese dalla rete) per le categoriche ad alta cardinalità, al posto del one-hot |
| Laboratorio 1 | Pipeline completa su un campione di Home Credit fornito come file pronto: Dataset e DataLoader di PyTorch, MLP con layer di embedding per le categoriche |
| Pillola 2 | Regolarizzazione contro l'overfitting: dropout, batch normalization, early stopping, weight decay. Gestione dello sbilanciamento: pesi di classe, focal loss (variante della cross-entropy che pesa di più gli esempi difficili) |
| Laboratorio 2 | Esperimenti controllati: una tecnica di regolarizzazione alla volta, lette sulle curve di apprendimento; confronto delle strategie anti-sbilanciamento sulle metriche di discriminazione |
| Pillola 3 | Calibrazione dell'output della rete: Platt scaling, regressione isotonica, temperature scaling. Cenno alle architetture tabellari dedicate (FT-Transformer, TabNet). Quando il deep learning ha senso sul tabellare: i tre regimi di dati delle tre strutture in aula — retail con milioni di righe, large corporate low-default, SME |
| Laboratorio 3 | Calibrazione del MLP; tabella finale champion/challenger MLP vs XGBoost su Gini, PR-AUC, Brier e test di calibrazione; discussione guidata sui portafogli low-default (pochi default osservati: stime del Gini instabili, test a bassa potenza) |

Dataset: campione di Home Credit Default Risk, fornito come file pronto (nessuna registrazione Kaggle in aula). Esercizi opzionali avanzati: embedding a dimensione variabile; confronto focal loss vs pesi di classe a parità di budget di training.

## Mattina 3 — Sequenze di transazioni: dove il deep learning vince

Obiettivo: il modulo distintivo del corso — passare dal feature engineering aggregato alle sequenze grezze di movimenti, dove le reti leggono il comportamento nel tempo che XGBoost non può vedere, e misurare quanto valore aggiungono.

| Blocco | Contenuto |
| --- | --- |
| Pillola 1 | Dal feature engineering aggregato (medie, conteggi, saldi su finestre temporali) alle sequenze grezze; rappresentazione di una sequenza di movimenti come tensore; padding e masking (allineamento di sequenze di lunghezza diversa e mascheramento delle posizioni fittizie) |
| Laboratorio 1 | Dal dataset sintetico di transazioni (conti correnti e carte) ai tensori: costruzione del dataset sequenziale con embedding delle causali, finestre temporali, DataLoader per sequenze |
| Pillola 2 | Reti per sequenze: RNN (reti ricorrenti), LSTM e GRU (varianti che gestiscono le dipendenze lunghe), reti convoluzionali 1D; cenno ad attention e Transformer |
| Laboratorio 2 | LSTM per la PD da sequenze di movimenti: training completo e valutazione con le metriche della mattina 1 |
| Pillola 3 | Quanto vale la sequenza: disegno dell'esperimento di confronto a parità di informazione; costi computazionali del training su sequenze e ponte verso l'appendice GPU |
| Laboratorio 3 | Confronto misurato a tre: LSTM su sequenze vs MLP su aggregati vs XGBoost su aggregati, con quantificazione del valore aggiunto in punti di Gini. Demo del docente dall'appendice GPU: lo stesso training su scheda grafica, con misura dello speedup |

Dataset: transazioni sintetiche costruite ad hoc per replicare la struttura dei dati della banca, con segnale comportamentale controllato. Esercizi opzionali avanzati: sostituire l'LSTM con una 1D-CNN; aggiungere un semplice meccanismo di attention.

## Mattina 4 — Esplicabilità, bias e governance

Obiettivo: spiegare le reti costruite nei giorni precedenti con Captum in continuità con lo SHAP già in uso, misurare e mitigare il bias, e tradurre tutto in documentazione pronta per la Vigilanza.

| Blocco | Contenuto |
| --- | --- |
| Pillola 1 | Da SHAP a Captum (la libreria di interpretabilità dell'ecosistema PyTorch): DeepSHAP e GradientSHAP (le varianti di Shapley per reti), Integrated Gradients (attribuzione che integra i gradienti da un input di riferimento all'input reale); spiegazioni globali vs locali; stabilità delle spiegazioni nel tempo e tra campioni |
| Laboratorio 1 | Captum sul MLP della mattina 2 con confronto diretto rispetto allo SHAP su XGBoost: stesse osservazioni, attribuzioni a confronto. Attribuzioni sulle sequenze dell'LSTM: quali movimenti pesano sulla PD |
| Pillola 2 | Vincoli di monotonicità (ad esempio: all'aumentare dell'utilizzo del fido la PD non può diminuire) e perché la Vigilanza li apprezza. Fairness quantitativa: demographic parity, equalized odds; strategie di mitigazione e loro costo |
| Laboratorio 2 | Audit di fairness con Fairlearn su un attributo protetto sintetico; applicazione di una mitigazione e misura del suo costo in performance |
| Pillola 3 | Governance del modello: PSI e monitoraggio della stabilità; ricalibrazione vs retraining vs challenger model — la traduzione pratica del "fine-tuning" chiesto dal supervisore; cosa documentare; cenno alla PD lifetime richiesta da IFRS 9 per i crediti in stage 2 |
| Laboratorio 3 | Esercizio di sintesi: compilazione della scheda di validazione del modello (template fornito) con le metriche, le spiegazioni e i test prodotti nelle quattro mattine; discussione finale (eventuali domande: al massimo 5 minuti) |

Esercizi opzionali avanzati: confronto della stabilità delle attribuzioni tra Integrated Gradients e DeepSHAP al variare del campione di riferimento.

## Appendice GPU

Tutti i laboratori del corso girano su CPU in pochi minuti; la GPU è trattata come capitolo a parte, con un notebook autonomo in tre parti:

1. verifica dell'ambiente: disponibilità di CUDA (la piattaforma NVIDIA che permette a PyTorch di usare la scheda grafica) e spostamento di tensori e modello sul device;
2. riesecuzione su GPU del training più pesante del corso — l'LSTM sulle sequenze della mattina 3 — con misura dello speedup (il rapporto tra tempo su CPU e tempo su GPU);
3. versione pronta per Google Colab, che offre gratuitamente una GPU T4: chi non dispone di una scheda locale replica lì l'esperimento.

Collocazione: fine della mattina 3, quando i tempi di addestramento cominciano davvero a pesare e il beneficio è tangibile. In aula il docente esegue la demo dal vivo sulla propria workstation; i partecipanti con GPU la replicano in locale, gli altri su Colab.

## Dataset e materiali consegnati

Tutti i dataset vengono distribuiti come file pronti dentro il materiale del corso: scaricare Home Credit da Kaggle richiederebbe account e autenticazione API, una fonte classica di perdite di tempo in aula.

| Materiale | Descrizione |
| --- | --- |
| environment.yml | La ricetta dell'ambiente conda: elenco dei pacchetti con versioni fissate (PyTorch CPU, scikit-learn, XGBoost, pandas, SHAP, Captum, Fairlearn, matplotlib), così tutti i partecipanti lavorano nello stesso ambiente |
| Notebook 0 — Setup | Verifica dell'ambiente da eseguire prima del corso (vedi sezione successiva) |
| 12 notebook di laboratorio | Tre per mattina, interamente commentati in italiano, con esercizi finali a difficoltà crescente |
| Notebook Appendice GPU | In doppia versione: locale (CUDA) e Google Colab |
| Slide delle pillole teoriche | In formato PDF, una per pillola |
| Dataset | Taiwan Credit Card Default (UCI); campione di Home Credit Default Risk; transazioni sintetiche di conto corrente e carte — tutti in formato CSV o Parquet, già etichettati secondo la definizione di default adottata |
| Template di scheda di validazione | Il documento compilato nell'esercizio di sintesi della mattina 4, riusabile internamente |
| Soluzioni degli esercizi | Consegnate a fine corso |

## Prerequisiti e verifica dell'ambiente

L'ambiente viene verificato insieme, in aula, in apertura della prima mattina di ciascuna edizione, con il Notebook 0 di setup.

- Prerequisiti dei partecipanti: conoscenza di base di Python, pratica con scikit-learn e con i modelli ad albero già in uso (XGBoost, Random Forest). Nessuna esperienza pregressa di deep learning richiesta.
- Notebook 0 — Setup: crea l'ambiente conda da environment.yml, verifica gli import di tutte le librerie ed esegue un mini-training di prova di circa 30 secondi; va riservato il tempo necessario in apertura della prima mattina.

Richieste al cliente prima dell'avvio: conferma delle ipotesi di lavoro della sezione dedicata; conferma che le postazioni possano creare ambienti conda e installare pacchetti da internet (proxy e permessi aziendali); calendario delle quattro mattine di ciascuna delle due edizioni — l'orario giornaliero è definito: 9:00–13:00.
