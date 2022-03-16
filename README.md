# SimCLR-TS-CSI

Proponiamo un framework per il rilevamento delle anomalie su dati sequenziali, in particolare su serie temporali. Il modello viene allenato in modo self-supervised utilizzando il contrastive learning. 

La rete è costruita sulla base del paper (poppelbaum),in cui viene proposta un'implementazione del modello SimCLR adattata alle  serie temporali, e del paper (csi), in cui viene proposta una variazione di SimCLR per migliorare la capacità della rete nel rilevamento delle anomalie. Il dataset utilizzato è TEP ed è lo stesso proposto nel paper (poppelbaum).

##### Struttura della rete

<figure align = "center"><img src="images\contrastive_learning.PNG" alt="Contrastive Learning" style="width:50%" ><figcaption align = "center"><p style="font-style: italic;">Fig.1 Contrastive Learning process</p></figcaption></figure>

La rete è molto semplice, consiste di tre livelli consecutivi Conv1d - LeakyReLU - BatchNorm1d (aggiungere parametri) e di un livello lineare terminale per fare la classificazione. Nell'implementazione originale di SimCLR la $f(.)$ è ResNet-50 in quanto lavora su dati di tipo immagine. Per adattare il modello alle serie temporali la $f(.)$ viene sostituita con un encoder costituito da livelli Con1d, questi livelli sono adatti poiché i dati temporali non posseggono informazioni spaziali tra i diversi canali.

Poiché non risultano evidenti vantaggi dall'uso della funzione non lineare $g(.)$, in (poppelbaum) si cerca direttamente di minimizzare la distanza tra le rappresentazioni latenti $h_i$ e $h_j$​ ottenute dall'encoder. Viene utilizzata una contrastive loss che segue la formula
$$
l_{i,j} = -log(\frac{exp(sim(h_i,h_j)/\tau)}{\sum_{k=1, k\neq i}^{2N}exp(sim(h_i,h_k)/\tau)})
$$
La struttura della rete è la seguente:

<figure align = "center"><img src="images\simclrts.PNG" alt="Contrastive Learning" style="width:50%" ><figcaption align = "center"><p style="font-style: italic;">Fig.2 SimCLR-TS</p></figcaption></figure>

##### Dataset

I dati provengono dal dataset TEP (ref), una raccolta di dati temporali composti da 52 canali con anomalie annotate, suddivisi in training e test secondo quanto suggerito in (poppelbaum). Il dataset contiene 22 classi, la classe 0 corrisponde al comportamento corretto e le restanti classi da 1 a 21 corrispondono a diversi comportamenti anomali. Il dataset di training è composto da 10560 campioni, mentre il dataset di training è composto da 17760 campioni. Entrambi i dataset sono costruiti in modo che le classi siano distribuite uniformemente e nessuna prevalga sulle altre. 

Tutti dati vengono normalizzati prima dell'uso, per la normalizzazione vengono usate la media e la varianza della porzione di training.

Per creare i sample da dare in input alla rete, vengono create delle finestre di ampiezza T=100 contenenti T campioni consecutivi. Di conseguenza ogni segnale in input ha una dimensione 52xT. Viene usato un approccio di tipo sliding window quindi, dato un segnale $s$, il segnale successivo inizia da $s+1$.

##### Trasformazioni (Data Augmentations)

Sono state selezionate sei trasformazioni dal paper di (poppelbaum) in particolare le tre che forniscono l'accuratezza migliore e le tre peggiori. In questo modo costruiamo due set di trasformazioni che chiameremo Soft Augmentations ( che contiene le trasformazioni con accuratezza migliore ) e Hard Augmentations ( contiene le altre). 

Soft Augmentations = { Left to Right, Crop Resize , Random Noise }

Left to Right : Il segnale viene capovolto lungo l'asse verticale tramite una moltiplicazione per matrice antidiagonale.

Crop Resize : Dal segnale originale si ottiene una copia dilatata nel tempo di un fattore 2, passando da un intervallo T ad uno 2T. I dati intermedi sono generati per interpolazione. Successivamente si campiona un istante nella prima metà del segnale generato(con periodo 2T) e si produce il segnale finale considerando i T istanti successivi. 
Nota : Con questa procedura avevamo delle prestazioni molto basse e il training diventava troppo lungo. Per un singolo batch di 64 fineste si impiegavano 6 minuti e qualche secondo. Abbiamo cambiato il modo di generare il segnale aumentato scambiando i passi di campionamento e generazione dei seguenti T istanti. Cosi facendo otteniamo lo stesso risultato ma con una riduzione del 45% del tempo di esecuzione dato che i cicli da eseguire per generare il segnale sono stati dimezzati. Il tempo di esecuzione di un batch è passato da 6 minuti a 3min 15 sec circa.  

Random Noise : Applica il rumore bianco al segnale aggiungendo (o sottraendo) la deviazione standard del segnale scalato da una variabile campionata da una distribuzione uniforme nel range (-1, 1).   

Hard Augmentations : { Blockout , Magnitude Warping, Permute Channels }

Blockout : Azzera una porzione del segnale a partire da un campione casuale.

Magnitude Warping : Somma una funzione sinusoidale al segnale con possibilita di cambiare frequenza e modulo.

Permute Channels : Permuta i canali del segnale.

##### Baseline

Nel paper (poppelbaum) i risultati vengono confrontati quelli ottenuti da un modello di baseline allenato in modo supervisionato. Per rendere paragonabili i risultati, il modello di baseline ha la stessa architettura del modello contrastive (3 livelli Conv1d - LeakyReLU - BatchNorm1d ) e viene allenato nella classificazione delle anomalie utilizzando una Cross Entropy Loss. Il training procede per 300 epoche.

Purtroppo i risultati che otteniamo da questo training sono molto lontani da quelli attesi. In (poppelbaum) si raggiunge un'accuratezza di circa 55% con la sola baseline, mentre noi otteniamo valori prossimi alla classificazione random (tra 0,5% e 0,7%). Come sanity check abbiamo cercato di mandare la rete in overfitting fornendo pochissimi dati di training (al massimo 20 batch da 64 sample) e testando sugli stessi dati. Ci saremmo aspettati un'accuratezza molto elevata, prossima al 100%, invece otteniamo valori che variano tra 30% e 50% in base a differenti configurazioni di learning rate e weight decay. Durante il training la loss scende correttamente ma evidentemente la rete non riesce ad estrarre correttamente le features.

##### Adattamento di CSI al modello