# config.py
# Dimensioni della superficie logica di simulazione (la finestra viene scalata per stare nello schermo)
WIDTH = 1500
HEIGHT = 1000
FPS = 60
WINDOW_SCREEN_FRACTION = 0.85   # frazione massima del desktop occupata dalla finestra all'avvio
SIM_SPEED_OPTIONS = (1, 2, 4, 8, 16)   # passi di simulazione per frame disegnato (tasti 1-5)
# Passo di simulazione fisso: tempi sul giro, timeout e reward NON dipendono dalle prestazioni della macchina.
SIM_DT_MS = 1000.0 / FPS

# Configurazione Simulazione MARL
CARS_PER_TEAM = 2
NUM_TEAMS = 10
TOTAL_CARS = CARS_PER_TEAM * NUM_TEAMS
TARGET_LAPS = 5
NUM_SIMULATIONS = 50
NUM_CHECKPOINTS = 50
SESSION_DURATION_MS = 180000  # durata minima della sessione (timeout)
# La sessione si allunga sui tracciati lunghi: deve bastare per TARGET_LAPS giri a questa velocità media.
# Con un limite fisso di 180 s, su un tracciato di 8487 px (giro teorico 37.5 s) nessuno poteva finire.
SESSION_MIN_AVERAGE_SPEED_PX = 150.0
POST_RACE_DELAY_MS = 2000
GRID_FRONT_GAP_PX = 10.0          # distanza della prima fila dalla linea di partenza
GRID_LATERAL_FRACTION = 0.45      # offset laterale delle due colonne della griglia (frazione della semi-larghezza)
AUTO_ADVANCE_DEFAULT = True   # avvio automatico della corsa successiva (attivabile/disattivabile con il tasto A)

# Algoritmi assegnati ai rispettivi Team
TEAM_ALGORITHMS = [
    "HEURISTIC",       # Team 0
    "GENETIC",         # Team 1
    "Q_LEARNING",      # Team 2
    "DQN",             # Team 3
    "POLICY_GRADIENT", # Team 4
    "GENETIC",         # Team 5
    "Q_LEARNING",      # Team 6
    "DQN",             # Team 7
    "POLICY_GRADIENT", # Team 8
    "HEURISTIC"        # Team 9
]

# Palette Colori Identificativi Team
TEAM_COLORS = [
    (230, 25, 75),   # Rosso
    (60, 180, 75),   # Verde
    (255, 225, 25),  # Giallo
    (0, 130, 200),   # Blu
    (245, 130, 48),  # Arancione
    (145, 30, 180),  # Viola
    (70, 240, 240),  # Ciano
    (240, 50, 230),  # Magenta
    (210, 245, 60),  # Lime
    (0, 128, 128)    # Verde Scuro
]

# --- Sensori (raggi di prossimità verso i muri) ---
# Cono frontale denso (-10/0/+10) per la frenata, diagonali per la sterzata, laterali corti per la
# posizione in carreggiata. Ogni raggio ha una portata propria, così anche i laterali (che vedono
# muri a ~30 px) sfruttano tutto l'intervallo [0, 1] invece di ~7% come con una portata unica di 400 px.
# Le distanze sono misurate dal bordo della carrozzeria, non dal centro dell'auto.
SENSOR_ANGLES = [-90, -50, -25, -10, 0, 10, 25, 50, 90]
SENSOR_RANGES = [120.0, 200.0, 300.0, 400.0, 400.0, 400.0, 300.0, 200.0, 120.0]
MAX_RAY_DISTANCE = max(SENSOR_RANGES)
SENSOR_STEP_PX = 2.0
CAR_HALF_LENGTH = 8.0
CAR_HALF_WIDTH = 5.0

# Punti di look-ahead sulla linea centrale (px di ascissa curvilinea davanti all'auto):
# vicino per la sterzata, medio/lontano per anticipare la frenata (spazio di frenata a v-max ~250 px).
LOOKAHEAD_DISTANCES_PX = (50.0, 150.0, 300.0)
TRAFFIC_RANGE_PX = 150.0
# Semicono anteriore: include le auto affiancate, necessarie per imparare a sorpassare.
TRAFFIC_CONE_DEG = 90.0

# Layout del vettore di osservazione (tutte le feature sono in [-1, 1]).
NUM_SENSORS = len(SENSOR_ANGLES)
OBS_SPEED = NUM_SENSORS
OBS_LATERAL = OBS_SPEED + 1          # offset laterale dalla mezzeria / semi-larghezza (+ = destra)
OBS_HEADING_SIN = OBS_LATERAL + 1    # errore di direzione rispetto alla tangente della pista
OBS_HEADING_COS = OBS_HEADING_SIN + 1
OBS_LOOKAHEAD = OBS_HEADING_COS + 1  # sin del bearing verso ciascun punto di look-ahead
# sin della direzione della pista nei punti di look-ahead rispetto alla direzione dell'auto: descrive la
# curva in arrivo indipendentemente dalla posizione laterale (utile per anticipare la frenata e il punto
# di corda e quindi per scoprire da soli la traiettoria ideale).
OBS_TRACK_HEADING = OBS_LOOKAHEAD + len(LOOKAHEAD_DISTANCES_PX)
OBS_TRAFFIC_PROXIMITY = OBS_TRACK_HEADING + len(LOOKAHEAD_DISTANCES_PX)
OBS_TRAFFIC_CLOSING = OBS_TRAFFIC_PROXIMITY + 1
OBS_TRAFFIC_BEARING = OBS_TRAFFIC_CLOSING + 1   # sin del bearing verso l'auto più vicina: da che lato passarla
OBSERVATION_SIZE = OBS_TRAFFIC_BEARING + 1
NUM_ACTIONS = 9

# Frequenza decisionale: i modelli che apprendono ripetono l'azione per N frame (20 Hz a 60 FPS).
# Decidere a 60 Hz rende l'esplorazione puro rumore e la credit assignment molto più difficile.
# L'euristico resta a 60 Hz perché è un controllore di riferimento, non un modello che apprende.
DEFAULT_DECISION_INTERVAL = 3
DECISION_INTERVAL_FRAMES = {"HEURISTIC": 1}

# --- Reward (tutte espresse in "checkpoint", quindi indipendenti dalla lunghezza del tracciato) ---
# Il progresso è potential-based (ascissa curvilinea): tornare indietro restituisce esattamente la
# reward incassata, quindi non esistono cicli sfruttabili e non c'è preferenza per stili di guida.
PROGRESS_REWARD_SCALE = 1.0          # reward per checkpoint di avanzamento (50 per giro)
# Una penalità costante per step, sommata a una terminazione, spinge a terminare presto ("suicide bias"):
# il discount già premia la velocità, quindi di default è disattivata.
TIME_PENALTY_PER_SECOND = 0.0
TERMINATION_PENALTY = 10.0           # identica per crash, stallo, contromano e assenza di progresso
COLLISION_PENALTY = 2.0              # solo all'auto responsabile (quella che tampona), a piena velocità d'impatto
# La penalità di collisione scala con la velocità d'impatto (velocità di avvicinamento) fino a questo valore:
# un contatto a passo d'uomo in griglia costa poco, un tamponamento a 150 px/s costa COLLISION_PENALTY.
# Con una penalità fissa, accelerare dietro un'auto ferma costava subito -2 e il PG ha imparato a restare fermo.
COLLISION_FULL_PENALTY_SPEED_PX = 150.0
FINISH_BONUS = 10.0
# La quota di reward del compagno dipende da uno stato non osservabile dall'agente e aggiunge rumore
# ai target TD; la cooperazione è già data dalla condivisione dei parametri tra le due auto del team.
TEAM_REWARD_SHARE = 0.0
# Bonus personal best: quando un'auto chiude un settore più velocemente del proprio record su quel tracciato
# riceve PERSONAL_BEST_REWARD_SCALE * miglioramento relativo (10% più veloce -> 1.0), con tetto
# PERSONAL_BEST_REWARD_CAP. È solo positivo, sparso e piccolo rispetto al progresso (~16 per settore) e
# alla terminazione (10): spinge a battere i propri tempi senza poter diventare la fonte principale di reward.
# Il primo passaggio in un settore non dà bonus (non c'è un riferimento). 0.0 per disattivarlo.
PERSONAL_BEST_REWARD_SCALE = 10.0
PERSONAL_BEST_REWARD_CAP = 1.5

# Traiettoria ideale: solo riferimento grafico (tasto L) e metro di valutazione nei log. Nessun agente,
# euristico compreso, la conosce: è il termine di paragone per capire se la scoprono da soli.
RACING_LINE_MARGIN_PX = 10.0         # distanza minima tra carrozzeria e muro sulla linea ideale
RACING_LINE_SPEED_MARGIN = 0.90      # frazione dei limiti di curvatura/frenata usata dal profilo di velocità

# Collisioni tra auto: non letali (rallentamento + penalità a chi tampona). Con collisioni letali la
# griglia di partenza eliminava ~70% delle auto, euristici compresi, in meno di un secondo.
LETHAL_CAR_COLLISIONS = False
CAR_COLLISION_DISTANCE = 12.0
COLLISION_COOLDOWN_FRAMES = 30
COLLISION_SPEED_FACTOR_AT_FAULT = 0.4
COLLISION_SPEED_FACTOR_VICTIM = 0.85

# Respawn di addestramento: un'auto eliminata resta fuori dalla classifica, ma per i modelli che
# apprendono dalle transizioni (Q-learning, DQN, Policy Gradient) un "fantasma" riparte poco prima
# del punto di uscita e continua a raccogliere esperienza fino a fine corsa. Il fantasma non interagisce
# con le altre auto (niente collisioni, scia o traffico) e non influisce su risultati o classifica.
TRAINING_RESPAWN = True
RESPAWN_DELAY_FRAMES = 30
RESPAWN_BACKOFF_PX = 40.0

# Squalifiche
STALL_SPEED_PX = 5.0
STALL_TIMEOUT_S = 3.0
WRONG_WAY_TIMEOUT_S = 2.5
NO_PROGRESS_TIMEOUT_S = 8.0          # deve avanzare di almeno un checkpoint in questo intervallo
# Penalità di squalifica "anticipata": mentre un'auto è ferma (o non avanza da più di NO_PROGRESS_GRACE_S)
# paga a ogni frame una quota della TERMINATION_PENALTY, e alla squalifica paga solo il resto. Il costo totale
# di una squalifica resta TERMINATION_PENALTY, ma diventa visibile subito: prima arrivava 3-8 s dopo, oltre
# l'orizzonte di gamma/GAE (peso ~0.01 dopo 3 s per il PG), e restare fermi sembrava gratis.
NO_PROGRESS_GRACE_S = 3.0

# Dinamica del Veicolo e Limiti di Tolleranza
MAX_STEER_ANGLE = 5.0       # Angolo massimo di sterzata per frame (gradi) a MIN_SPEED_PX
MAX_SPEED_PX = 360.0       # Velocità massima nei rettilinei (px/s)
MIN_SPEED_PX = 80.0       # Sotto questa velocità il raggio di curvatura resta costante (no rotazione da fermi)
ACCELERATION = 180.0       # Rateo incremento velocità (px/s^2)
DECELERATION = 260.0       # Rateo decremento velocità (px/s^2)
COASTING_DECELERATION = 25.0
SCRUB_DECELERATION = 90.0  # perdita graduale di velocità sterzando oltre l'85% della v-max
