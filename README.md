# Institutional Quant AI: Trend-Boosted Edition

Un sistem hibrid de tranzactionare cantitativa bazat pe Deep Learning (Bidirectional LSTM) si strategii institutionale de management al riscului. 

Acest proiect reprezinta lucrarea de licenta si demonstreaza capacitatea de a extrage semnale financiare din medii cu zgomot ridicat folosind arhitecturi de inteligenta artificiala, augmentate de reguli matematice stricte de conservare a capitalului. Interfata grafica este construita in Streamlit.

---

## Arhitectura Algoritmului de Predictie

Sistemul implementeaza o conducta cantitativa (Quant Pipeline) impartita in 4 etape fundamentale:

### Etapa 1: Ingestia si Ingineria Trasaturilor (Feature Engineering)
Datele brute prelucrate via `yfinance` sunt transformate in trasaturi financiare cu relevanta institutionala:
* **Log Returns (Randamente Logaritmice):** Folosite in detrimentul randamentelor procentuale simple pentru a asigura aditivitatea in timp si simetria statistica, forțând distributia datelor sa se apropie de o distributie normala.
* **Garman-Klass Volatility:** Un estimator de volatilitate continua care capteaza miscarile intraday (folosind Open, High, Low, Close), mult mai precis decat deviatia standard clasica.
* **Chaikin Money Flow (CMF):** Un indicator de volum care semnaleaza presiunea de acumulare/distributie a banilor institutionali.
* **Codificarea Sezonalitatii (Day_Sin / Day_Cos):** Maparea zilelor saptamanii in functii trigonometrice continue pentru a ajuta reteaua sa identifice tipare temporale recurente.

### Etapa 2: Preprocesarea Temporala (Time-Series Prep)
* **Robust Scaler:** Datele sunt normalizate ignorand valorile aberante (outliers/crash-uri bursiere extreme) pentru a nu distorsiona invatarea retelei.
* **Fereastra de Timp (Sliding Window):** Datele sunt transformate in tensori tridimensionali. Modelul analizeaza secvente de 60 de zile pentru a prezice ziua urmatoare.
* **Exponential Decay Weights:** Sistemul asigneaza o pondere matematica mai mare datelor recente in defavoarea datelor vechi, permitand modelului sa se adapteze la schimbarile de regim ale pietei.

### Etapa 3: Arhitectura Deep Learning (Bagged Attention LSTM)
Motorul central este un ansamblu de 3 retele neurale antrenate separat. Arhitectura fiecarei retele include:
1. **Gaussian Noise:** Adaugarea controlata de zgomot pe datele de intrare pentru a preveni fenomenul de overfitting si a forta reteaua sa generalizeze.
2. **Bidirectional LSTM:** Un strat recurent care proceseaza fereastra de timp din ambele directii, extragand corelatii complexe invizibile unei retele standard. Memoria interna (Cell State) rezolva problema gradientului descrescator.
3. **Custom Attention Layer:** Un mecanism care asigneaza scoruri de importanta zilelor din fereastra de timp, permitand modelului sa se concentreze exclusiv pe zilele cu socuri de volatilitate sau semnale clare.
4. **Ensemble Aggregation:** Predictia finala este media predictiilor celor 3 modele din ansamblu, reducand dispersia si crescand acuratetea directionala.

### Etapa 4: Logica de Executie si Managementul Riscului (The Risk Engine)
Predictia inteligentei artificiale trece printr-un strat de inginerie financiara inainte de a fi transformata intr-o pozitie de tranzactionare:
* **Trend Filtering & Signal Boosting:** Daca activul este deasupra mediei mobile de 50 de zile (SMA-50), sistemul aplica un multiplicator de incredere predictiilor de crestere.
* **Volatility Targeting:** Sistemul calculeaza frica din piata (prin Garman-Klass). Daca volatilitatea pietei depaseste tinta setata, algoritmul reduce automat dimensiunea pozitiei tranzactionate, protejand capitalul.
* **Sigmoid Conviction Sizing:** Functia matematica Sigmoid converteste predictia modelului intr-un scor de convingere intre 0 si 100%, dictand marimea alocarii de capital.
* **Long-Only Safety Switch:** O regula hard-coded care interzice pariurile de vanzare in lipsa (Shorting). Daca modelul prezice o scadere a pietei, algoritmul vinde si se refugiaza in 100% Cash.

---

## Tehnologii Utilizate
* **Python 3.10+**
* **TensorFlow / Keras:** Constructia si antrenarea arhitecturii Deep Learning.
* **Scikit-Learn:** Scalarea datelor (RobustScaler) si preprocesare.
* **Pandas & NumPy:** Manipularea datelor si calcule matriciale vectorizate.
* **Streamlit:** Constructia interfetei web.
* **YFinance:** API-ul de extragere a datelor istorice de piata.

---

## Instalare si Rulare (Local)
git clone [https://github.com/username/numele-proiectului.git](https://github.com/username/numele-proiectului.git)
cd numele-proiectului
pip install -r requirements.txt
streamlit run app.py

Interpretarea Metricilor de Performanta

Sistemul afiseaza 5 metrici vitale post-backtest (calculate net de comisioane):

    Net Profit: Randamentul absolut obtinut la finalul perioadei.

    Sharpe Ratio: Raportul dintre randament si volatilitate (riscul asumat pe unitatea de profit).

    Win Rate: Rata de succes a tranzactiilor individuale.

    Max Drawdown: Riscul maxim de ruina (scaderea procentuala maxima de la ultimul varf istoric al portofoliului).

    Realized Accuracy: Procentajul in care decizia de directie a AI-ului a fost corecta si suficient de puternica pentru a acoperi costurile de tranzactionare.
