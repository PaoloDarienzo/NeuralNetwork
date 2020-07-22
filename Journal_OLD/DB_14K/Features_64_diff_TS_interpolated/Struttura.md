Nelle due cartelle qui presenti, vengono analizzati modelli
con differenti dimensioni delle immagini.

Tutte le immagini vengono interpolate.

In Bi_CuDNN viene applicato il modello Bidirezionale CuDNNLSTM,
sulle dimensioni:
1.	64x64, 88,6%, 62 epoche.
2.	128x64, 92%, 80 epoche.
3.	256x64, 90,36%, 57 epoche.
	
In Bi_CONV2D_LSTM, viene applicato il modello Bidirezionale CONV2D_LSTM,
sulle dimensioni:
1.	(64, 8, 8, 1), troppo pesante da portare a termine.
2.	(16, 16, 16, 1), 89,1%, 19 epoche.
3.	(4, 32, 32, 1), 84,6%, 13 epoche.
4.	(8, 8, 8, 8), 89,33%, 31 epoche.