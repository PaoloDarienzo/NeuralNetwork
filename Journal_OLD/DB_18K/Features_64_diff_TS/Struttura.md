Bi_CudNN_18k_64x64, 84,44%, 99 epoche.

Bi_CudNN_18k_128x64, 90,56%, 100 epoche.

Bi_CudNN_18k_256x64, 86,80%, 100 epoche.

===

Tutte da immagini 128x64.
Bi_CONV2D_LSTM_18k(16, 8, 8, 8), 91,36%, 32 epoche.
Bi_CONV2D_LSTM_18k(8, 32, 32, 1), 90,53%, 13 epoche.
Bi_CONV2D_LSTM_18k(32, 16, 16, 1), 90,92%, 20 epoche.


Per i modelli LSTM Convoluzionali (CONV2D_STLM), 
c'è bisogno di una manipolazione delle dimensioni delle immagini particolare: 
siccome la convoluzione LSTM necessita di immagini da analizzare nel tempo correlate,
le ricavo da una singola immagine analizzata, ossia per ogni immagine creo delle sotto-immagini,
su cui posso applicare la convoluzione legata ai time steps.
Posso mantenere il terzo parametro delle immagini, i channels, come 1, oppure
posso sfruttarlo per rimodellare il set di immagini.

Le immagini vengono lette e poi 0-padded in una voluta dimensione.
Opero in funzione della dimensione delle sotto-immagini e sotto condizione della RAM.

Se voglio avere sotto-immagini del tipo (8, 8, 8), posso sfruttare 16 time steps.
Per avere (16, 8, 8, 8), posso fare il crop ad immagini (128, 64).

Se voglio sotto-immagini del tipo (32, 32, 1), posso sfruttare 8 time steps.
Per avere (8, 32, 32, 1), posso fare il crop ad immagini (128, 64) (e ridurre la dimensione del batch).

Se voglio sotto-immagini del tipo (16, 16, 1), posso sfruttare 32 time steps.
Per avere (32, 16, 16, 1), , posso fare il crop ad immagini (128, 64).