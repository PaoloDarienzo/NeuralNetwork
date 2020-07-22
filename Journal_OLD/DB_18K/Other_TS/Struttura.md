Tutti i notebook implementano il modello Bi_CuDNN_LSTM.

Vengono testati diversi modelli che implementano
diversi time steps: 32, 64, 128, 256, 512.
Il file "Bi_CudNN_18k_512x64_flatten_TS_extreme" testa
i valori di time steps come 1, 2, 4, 8, 16.
Il file "Bi_CudNN_18k_TS_extreme" testa 1 time steps e (64*64)/4 time steps.

Questi file sono da controllare al riguardo della lista 
di offuscamenti erronei.


Nella cartella "Truncated_images" sono utilizzate immagini 0-padded
nei formati 64x64, 128x32, 128x64, 256x32, 256x64.

Tutte le immagini usate nei seguenti notebook, sono interpolate.

Bi_CuDNN_18k_32x128:
1.   2 layer da 141 nodi, 94 nodi;
2.   Immagini 32x128;
3.   1024 batch size;
4.   Accuratezza finale: 91%.

Bi_CudNN_18k_64x128.
1.   2 layer da 141 nodi, 94 nodi;
2.   Immagini 64x128;
3.   1024 batch size;
4.   Accuratezza finale: 91,5%.

Bi_CudNN_18k_128x32:
1.   2 layer da 141 nodi, 94 nodi;
2.   Immagini 128x32;
3.   1024 batch size;
4.   Accuratezza finale: 80%.

Bi_CudNN_18k_256x32:
1.   2 layer da 141 nodi, 94 nodi;
2.   Immagini 256x32;
3.   1024 batch size;
4.   Accuratezza finale: 90%.