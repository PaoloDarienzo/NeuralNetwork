Nella cartella "Features_64_diff_TS" ci sono i notebook
che hanno come time_steps diversi valori e fisso a 64
il numero di features (2 tipi di modelli).

Nella cartella "format_64x596-596x64" ci sono i modelli Bi_CudNN
che utilizzano le immagini nel suddetto formato;
tutte le immagini sono 0-padded.

Nella cartella "Other_TS" ci sono modelli che implementano
diversi time steps: 32, 64, 128, 256, 512.
Il file "Bi_CudNN_18k_512x64_flatten_TS_extreme" testa
i valori di time steps come 1, 2, 4, 8, 16.
Il file "Bi_CudNN_18k_TS_extreme" testa 1 time steps e (64*64)/4 time steps.
Alcuni notebook implementano le immagini interpolate, altri le immagini croppate.

Nella cartella "Vanilla NN" sono implementate le reti neurali shallow
con immagini 64x64 interpolate e croppate, e con immagini 596x64 croppate.