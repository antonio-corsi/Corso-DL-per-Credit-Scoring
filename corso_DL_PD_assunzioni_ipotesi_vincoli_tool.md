# Corso "Deep Learning per il Rischio di Credito: stima della PD con PyTorch"
## Assunzioni, ipotesi di lavoro, vincoli e tool

**Stato**: documento di riferimento interno — aggiornato al 4 novembre 2026.
**Committente**: Intesa Sanpaolo, tramite Overnet/Digit'Ed — esigenze dal verbale Digit'Ed/ISP del 15 settembre 2026.
**Scopo**: raccogliere in un unico punto tutto ciò su cui il corso poggia, distinguendo ciò che il cliente ha imposto (vincoli), ciò che abbiamo deciso noi e già comunicato (assunzioni), ciò che abbiamo adottato in mancanza di indicazioni e che va validato (ipotesi di lavoro) e lo stack tecnico (tool).

---

## 1. Assunzioni

Scelte di impostazione adottate da noi, già comunicate al cliente o incorporate nel programma di dettaglio; modificabili su richiesta con impatto limitato.

| # | Assunzione | Nota |
|---|-----------|------|
| A1 | Orario **9:00–13:00**, con un intervallo di 15 minuti a metà mattina | 240 minuti lordi, 225 netti di lezione per mattina: coerente con la V1 di questo documento (3 ore e 45 minuti effettive) |
| A2 | Schema ricorrente di ogni mattina, **senza minutaggi prefissati**: apertura → tre pillole teoriche brevi, ciascuna seguita dal proprio laboratorio guidato → intervallo a metà mattina → sintesi finale | Il rapporto teoria–pratica ≈ 30–70 si realizza sull'arco dell'intera mattina; i tempi dei singoli blocchi restano liberi, da adattare all'andamento dell'aula. **Nessun blocco dedicato alle domande**: la platea ne pone molto poche — al massimo 5 minuti in chiusura, il tempo resta ai laboratori |
| A3 | **Nessun dato reale della banca in aula**: si lavora su dataset pubblici e sintetici che ne replicano la struttura (transazioni di conto corrente e carte) | Assunzione prudenziale per riservatezza; se il cliente mette a disposizione dati anonimizzati, i notebook sono adattabili |
| A4 | Obiettivo modellistico: **PD binaria a orizzonte fisso** (default sì/no entro 12 mesi dalla data di osservazione) — classificazione supervisionata con output probabilistico | Il cliente **non si è espresso** sulla forma del target: l'assunzione discende (a) dall'attività quotidiana dichiarata dal cliente — modelli di PD costruiti con classificatori come XGBoost, Random Forest e alberi (vedi V7–V8), che trattano il problema come classificazione binaria con output probabilistico — e (b) dallo standard IRB della PD a 12 mesi (I5). Alternative come i modelli di sopravvivenza (*survival analysis*: stima del tempo al default anziché del sì/no a orizzonte fisso) o la PD multi-periodo restano fuori perimetro, salvo diversa indicazione |
| A5 | XGBoost presente **in ogni laboratorio come benchmark** (approccio *champion/challenger*: modello in produzione contro modello sfidante), mai come oggetto di insegnamento | Metodologia corretta e risposta alla domanda tipica della Vigilanza ("perché una rete invece di XGBoost?"); la letteratura mostra che sui dati tabellari il gradient boosting resta un benchmark severo per il deep learning: Grinsztajn et al. (2022), Shwartz-Ziv & Armon (2022), Borisov et al. (2024) — riferimenti completi in §5 |
| A6 | Esercizi di fine lab **a difficoltà crescente**, con estensioni opzionali per i partecipanti con Python medio-alto | Gestione dell'eterogeneità dei livelli dichiarata dal cliente |
| A7 | Dataset distribuiti come **file pronti (CSV/Parquet)** dentro il materiale del corso | Evita registrazioni e autenticazioni Kaggle in aula (fonte classica di perdite di tempo) |
| A8 | Terminologia: italiano con il **termine inglese tra parentesi** alla prima occorrenza di ogni concetto tecnico | Letteratura, librerie e messaggi d'errore sono in inglese |

---

## 2. Ipotesi di lavoro (da validare dal cliente)

Punti su cui il cliente non ha fornito indicazioni (risposte 5 e 6): adottiamo la prassi europea corrente, dichiarandola in aula come tale ("ipotesi standard di prassi, non valori imposti da una norma"). La **funzione di validazione interna** della banca deve confermarle prima dell'avvio; la sostituzione è un intervento minimo sui materiali.

### 2.1 Soglie e test di Vigilanza (risposta 5: "usa ipotesi ragionevoli")

| # | Ambito | Ipotesi adottata |
|---|--------|------------------|
| I1 | Discriminazione | **Gini** (= 2·AUC − 1) come metrica principale, con lo schema "a semaforo" di prassi: sotto 40% debole, 40–60% accettabile, sopra 60% buono. Avvertenza esplicita per i portafogli *low-default* (pochi default osservati, es. large corporate): stime del Gini instabili, test a bassa potenza |
| I2 | Stabilità | **PSI** (*Population Stability Index*, indice dello spostamento della popolazione corrente rispetto a quella di sviluppo) con soglie classiche: 0,10 (stabile) e 0,25 (deriva significativa) |
| I3 | Calibrazione | **Test binomiale** e **test di Jeffreys** per fascia di rating — quest'ultimo previsto dalle istruzioni BCE per il *validation reporting* dei modelli interni — più Hosmer–Lemeshow come alternativa classica |

### 2.2 Definizione di default e orizzonte della PD (risposta 6: "usa ipotesi ragionevoli")

| # | Ambito | Ipotesi adottata |
|---|--------|------------------|
| I4 | Definizione di default | Quella regolamentare europea dell'**art. 178 CRR** (*Capital Requirements Regulation*, il regolamento europeo sui requisiti di capitale): 90 giorni di arretrato rilevante (*past due*) **oppure** improbabile adempimento (*unlikely to pay*, UTP), con le soglie di rilevanza EBA: 100 € per il retail, 500 € per il non retail, più l'1% relativo dell'esposizione |
| I5 | Orizzonte della PD | **12 mesi**, target binario "default entro 12 mesi dalla data di osservazione", standard dei modelli **IRB** (*Internal Ratings-Based*, i modelli interni autorizzati per il calcolo del capitale). Cenno in teoria alla PD *lifetime* richiesta da IFRS 9 per i crediti in stage 2 |
| I6 | Etichettatura dei dataset | I dataset sintetici vengono etichettati secondo I4–I5, così il vocabolario del corso coincide con quello usato dai partecipanti nei confronti della Vigilanza |

### 2.3 Quadro normativo citato nei materiali

- CRR (*Capital Requirements Regulation*);
- linee guida EBA sulla stima della PD: **EBA/GL/2017/16** (EBA = Autorità Bancaria Europea);
- istruzioni **BCE** sul *validation reporting* dei modelli interni.

---

## 3. Vincoli (requisiti del cliente, non negoziabili)

### 3.1 Formato e logistica

| # | Vincolo |
|---|---------|
| V1 | Corso **online**, **4 mattine piene** da **3 ore e 45 minuti** ciascuna (≈ 15 ore totali), erogato in **2 edizioni identiche** |
| V2 | Rapporto teoria–pratica **30–70** |
| V3 | Lingua dei materiali: **italiano** |

### 3.2 Platea

| # | Vincolo |
|---|---------|
| V4 | **15 partecipanti per edizione** (2 edizioni identiche, 30 persone in totale), tutti con laurea STEM e profilo di **analista quantitativo** |
| V5 | Conoscenza di Python **di base** (alcuni medio-alta) |
| V6 | Tre strutture della banca con regimi di dati diversi: **retail** (big data, ~10⁶ righe), **large corporate** (anche banche ed enti pubblici: moli inferiori, portafogli contenuti), **SME retail/corporate** (moli inferiori) |

### 3.3 Competenze pregresse e perimetro

| # | Vincolo |
|---|---------|
| V7 | Attività quotidiana: costruzione di modelli di **PD** e *pattern recognition* (quali predittori spiegano meglio la risposta) su transazioni di conto corrente e carte di credito, a grandissimi volumi |
| V8 | Tecniche già padroneggiate con scikit-learn — **XGBoost, Random Forest, alberi decisionali** — che **non sono oggetto del corso** (solo benchmark di confronto) |
| V9 | Metriche oggi in uso: **solo accuracy** (F1, precision, recall non note) → il corso parte da qui |
| V10 | Interpretabilità oggi in uso: **Shapley e feature importance** → punto di aggancio per Captum |
| V11 | Obiettivo del cliente: costruire modelli predittivi di PD **anche con reti neurali**; forte interesse per **interpretabilità ed esplicabilità** delle reti |

### 3.4 Contesto regolamentare

| # | Vincolo |
|---|---------|
| V12 | Soggetti a **controlli attenti della Vigilanza**: severa, composta da tecnici; attenzione particolare a **metriche di performance**, **metriche di interpretabilità/esplicabilità** e **bias** |
| V13 | La Vigilanza a volte chiede **fine-tuning dei modelli** → nel corso tradotto in pratica: ricalibrazione vs retraining vs challenger model |

### 3.5 Tecnologia e ambiente

| # | Vincolo |
|---|---------|
| V14 | Stack concordato: **Python, scikit-learn, PyTorch** |
| V15 | La banca lavora su **Google Cloud** (contesto; i lab non lo richiedono) |
| V16 | Postazioni: **Visual Studio Code**, **ambienti conda**, **internet accessibile** |
| V17 | **GPU in capitolo a parte**: demo per chi ne dispone, oppure su **Google Colab** — tutti i lab del corso devono girare su CPU |

---

## 4. Tool del corso

### 4.1 Librerie Python (fissate in `environment.yml`)

| Libreria | Ruolo nel corso |
|----------|-----------------|
| **PyTorch** (build CPU) | Framework di deep learning: MLP tabellari, embedding, LSTM su sequenze |
| **scikit-learn** | Preprocessing, regressione logistica (baseline), metriche, calibrazione |
| **XGBoost** | Benchmark/challenger in ogni laboratorio |
| **pandas** | Manipolazione dei dati tabellari e delle sequenze |
| **SHAP** | Interpretabilità del champion XGBoost (continuità con la pratica attuale) |
| **Captum** | Interpretabilità delle reti PyTorch: DeepSHAP, GradientSHAP, Integrated Gradients |
| **Fairlearn** | Metriche di fairness (demographic parity, equalized odds) e mitigazioni |
| **matplotlib** | Grafici: curve ROC, curve di calibrazione, curve di apprendimento |

`environment.yml` = la "ricetta" dell'ambiente conda: elenco dei pacchetti con **versioni fissate**, così tutti i partecipanti di entrambe le edizioni lavorano nello stesso ambiente.

### 4.2 Strumenti di lavoro

| Strumento | Uso |
|-----------|-----|
| **Visual Studio Code** | IDE di riferimento per l'esecuzione dei notebook |
| **conda** | Gestione dell'ambiente (creazione da `environment.yml`) |
| **Jupyter notebook** | Formato di tutti i laboratori (notebook interamente commentati in italiano) |
| **Google Colab** | Replica dell'appendice GPU per chi non dispone di scheda locale (GPU T4 gratuita) |
| **CUDA** | Piattaforma NVIDIA che permette a PyTorch di usare la GPU — solo nell'appendice dedicata |

### 4.3 Dataset

| Dataset | Mattine | Note |
|---------|---------|------|
| **Taiwan Credit Card Default** (UCI) | 1 | Storico pagamenti mensili; ideale per introdurre metriche e primo MLP |
| **Home Credit Default Risk** (campione) | 2 | Ricco di variabili comportamentali e categoriche ad alta cardinalità; fornito come file pronto (nessuna registrazione Kaggle) |
| **Transazioni sintetiche** (costruite ad hoc) | 3–4 | Replicano la struttura di conti correnti e carte della banca, con segnale comportamentale controllato; etichettate secondo le ipotesi I4–I5 |

### 4.4 Materiali consegnati

- `environment.yml`;
- Notebook 0 — Setup (verifica dell'ambiente: creazione da `environment.yml`, controllo degli import, mini-training di prova — da eseguire in apertura della prima mattina);
- 12 notebook di laboratorio (3 per mattina) + notebook Appendice GPU in doppia versione (locale CUDA / Colab);
- slide delle pillole teoriche (PDF);
- dataset in CSV/Parquet;
- template di **scheda di validazione del modello** (compilato nell'esercizio di sintesi della mattina 4, riusabile internamente);
- soluzioni degli esercizi (a fine corso).

---

## 5. Riferimenti bibliografici

A supporto dell'assunzione A5 (XGBoost come benchmark in ogni laboratorio):

- Grinsztajn L., Oyallon E., Varoquaux G. (2022), *"Why do tree-based models still outperform deep learning on typical tabular data?"*, Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track — arXiv:2207.08815. Benchmark su 45 dataset: sui tabellari di dimensioni medie (~10.000 campioni) i modelli ad albero restano lo stato dell'arte;
- Shwartz-Ziv R., Armon A. (2022), *"Tabular data: Deep learning is not all you need"*, Information Fusion, 81, 84–90 — arXiv:2106.03253. A parità di protocollo di tuning, XGBoost batte i modelli deep proposti per il tabellare sulla maggior parte dei dataset; l'ensemble deep + XGBoost fa meglio di XGBoost da solo;
- Borisov V., Leemann T., Seßler K., Haug J., Pawelczyk M., Kasneci G. (2024), *"Deep Neural Networks and Tabular Data: A Survey"*, IEEE Transactions on Neural Networks and Learning Systems, 35(6), 7499–7519 (online dal 2022), DOI 10.1109/TNNLS.2022.3229161 — arXiv:2110.01889. Rassegna sistematica: gli ensemble a gradient boosting restano per lo più superiori ai modelli deep nei compiti supervisionati su tabellare.

I tre lavori motivano con onestà intellettuale il ruolo di XGBoost come benchmark: il valore delle reti va cercato sui grandi volumi e, soprattutto, sulle **sequenze grezze di transazioni**, fuori dal perimetro di quei benchmark.

---

## 6. Punti ancora aperti con il cliente

- [ ] Validazione delle ipotesi I1–I6 da parte della funzione di validazione interna;
- [ ] conferma che le postazioni possano creare ambienti conda e installare pacchetti da internet (proxy/permessi aziendali);
- [ ] calendario delle quattro mattine di ciascuna delle due edizioni (l'orario giornaliero è definito: 9:00–13:00, v. A1);
- [ ] tempo da riservare in apertura della prima mattina per il Notebook 0 di setup (la verifica pre-corso non è fattibile);
- [ ] correzione della slide di presentazione Overnet: la riga "didattica frontale / senza installazioni / demo sul PC del docente" va allineata alla formula hands-on concordata (v. V2, V16) — testo sostitutivo già fornito.
