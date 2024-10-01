from abc import ABC, abstractmethod
import numpy as np


class StorageSystem(ABC):
    """
    Classe astratta che rappresenta un sistema di accumulo di energia.

    Questa classe definisce le interfacce per le operazioni di base che qualsiasi sistema di accumulo deve implementare,
    come la carica, il reset e la terminazione.

    Metodi astratti:
        charge(energy: float, production: float) -> float:
            Metodo per caricare o scaricare energia dal sistema di accumulo.

        reset() -> None:
            Ripristina lo stato iniziale del sistema di accumulo.

        terminate() -> None:
            Termina e pulisce eventuali risorse o stati finali del sistema di accumulo.
    """

    @abstractmethod
    def charge(self, energy: float, production: float) -> float:
        """
        Metodo astratto per caricare o scaricare energia dal sistema di accumulo.
        Dev'essere implementato dalle classi derivate.

        Args:
            energy (float): L'energia da caricare o scaricare.
            production (float): Produzione di energia disponibile per la carica.

        Returns:
            float: Violazione dei vincoli di carica o scarica (se presente).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Ripristina lo stato iniziale del sistema di accumulo.
        """
        pass

    @abstractmethod
    def terminate(self) -> None:
        """
        Termina l'operazione del sistema di accumulo e libera eventuali risorse.
        """
        pass


class Battery(StorageSystem):
    """
    Classe che rappresenta una batteria come sistema di accumulo di energia.

    Gestisce la carica e la scarica della batteria, applicando vincoli fisici come capacità massima,
    potenza nominale e perdite di efficienza. La batteria subisce degradazione della capacità nel tempo
    e tiene traccia delle violazioni dei vincoli durante le operazioni di carica e scarica.

    Attributes:
        capacity (float): Capacità massima della batteria (kWh).
        rated_power (float): Potenza nominale della batteria (kW).
        soc_min (float): Stato minimo di carica (SoC) della batteria (kWh).
        ETA (float): Efficienza di carica/scarica della batteria.
        ALFA (float): Coefficiente di perdita per il calcolo del SoC.
        BETA (float): Coefficiente di degradazione della batteria.
        soc (float): Stato attuale di carica della batteria (kWh).
        time_step (int): Il contatore del tempo per il monitoraggio degli step di simulazione.
        soc_init_list (List[float]): Lista che memorizza lo stato iniziale di carica per ogni step.
        soc_final_list (List[float]): Lista che memorizza lo stato finale di carica per ogni step.
        energy_desired (List[float]): Lista delle energie desiderate durante la carica/scarica.
        actual_charges (List[float]): Lista delle quantità effettivamente caricate.
        actual_discharges (List[float]): Lista delle quantità effettivamente scaricate.
        constraint_violation (List[float]): Lista delle violazioni dei vincoli di carica/scarica.
    """

    def __init__(self, capacity: float, rated_power: float, soc_min: float, efficiency: float, alfa: float,
                 beta: float):
        """
        Inizializza una nuova batteria con i parametri specificati.

        Args:
            capacity (float): Capacità massima della batteria (kWh).
            rated_power (float): Potenza nominale della batteria (kW).
            soc_min (float): Stato minimo di carica (SoC) come percentuale della capacità totale.
            efficiency (float): Efficienza della batteria (valore tra 0 e 1).
            alfa (float): Coefficiente di perdita per la carica/scarica.
            beta (float): Coefficiente di degradazione della capacità.
        """
        self.capacity = capacity
        self.c0 = capacity
        self.rated_power = rated_power
        self.soc_min = soc_min * capacity
        self.ETA = efficiency
        self.ALFA = alfa
        self.BETA = beta

        # Attributi che verranno resettati
        self.soc = self.soc_min
        self.time_step = 0
        self.soc_init_list, self.soc_final_list = [], []
        self.energy_desired, self.actual_charges, self.actual_discharges = [], [], []
        self.constraint_violation = []

    def charge(self, energy: float, production: float) -> float:
        """
        Gestisce sia la carica che la scarica della batteria in base ai vincoli fisici e alle condizioni di produzione.

        Se l'energia è positiva, si tenta di caricare la batteria. Se è negativa, si tenta di scaricarla.
        La funzione rispetta i vincoli di capacità, potenza nominale, efficienza e arbitraggio energetico.

        Args:
            energy (float): Energia da caricare (> 0) o scaricare (< 0) nella batteria.
            production (float): Produzione disponibile da cui caricare la batteria (per vincolo di arbitraggio).

        Returns:
            float: Violazione dei vincoli di carica o scarica della batteria.
        """
        self.energy_desired.append(energy)
        self.soc_init_list.append(self.soc)

        # Calcolo del nuovo SoC (State of Charge)
        soc_init = self.soc * (1 - self.ALFA)

        # Carica della batteria
        if energy > 0:
            # Vincolo della potenza nominale
            energy = min(energy, self.rated_power)
            # Vincolo della capacità
            energy = min(energy, self.capacity - self.soc)
            # Vincolo arbitraggio energetico
            charging_energy = min(energy, production)
            # Aggiornamento del SoC
            self.soc = soc_init + charging_energy * self.ETA
            discharging_energy = 0.
        else:
            # Vincolo della potenza nominale per la scarica
            energy = max(energy, -self.rated_power)
            # Vincolo della capacità per la scarica
            discharging_energy = max(energy, min(self.soc_min - self.soc, 0))
            # Aggiornamento del SoC
            self.soc = soc_init + discharging_energy / self.ETA
            charging_energy = 0.

        self.actual_charges.append(charging_energy)
        self.actual_discharges.append(-discharging_energy)

        constrained_energy = charging_energy + discharging_energy
        violation = np.abs(constrained_energy - energy)
        violation = np.round(violation, 5)
        self.constraint_violation.append(violation)

        # Controllo del vincolo del SoC
        assert 0. <= self.soc <= self.capacity, (
            f'SoC: {self.soc:.2f} non valido.\n'
            f'Capacità massima: {self.capacity}, SoC minimo: {self.soc_min}\n'
            f'SoC iniziale: {soc_init:.2f}, Energia: {constrained_energy:.2f}')
        self.soc_final_list.append(self.soc)

        # Energia scambiata con l'ambiente
        if constrained_energy > 0:
            energy_trade = (self.soc - soc_init) / self.ETA
        else:
            energy_trade = (self.soc - soc_init) * self.ETA

        # Degradamento della capacità della batteria
        try:
            self.capacity -= self.BETA * self.c0 * np.abs(energy_trade) / (2 * self.capacity)
        except ZeroDivisionError:
            self.capacity = 0.

        return violation

    def reset(self) -> None:
        """
        Ripristina lo stato iniziale della batteria, resettando tutti gli attributi variabili.
        """
        self.soc = self.soc_min
        self.time_step = 0
        self.soc_init_list, self.soc_final_list = [], []
        self.energy_desired, self.actual_charges, self.actual_discharges = [], [], []
        self.constraint_violation = []

    def terminate(self) -> None:
        """
        Termina le operazioni della batteria. Questo metodo può essere usato per liberare risorse
        o salvare lo stato finale, se necessario.
        """
        pass
