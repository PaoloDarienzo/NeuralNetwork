Nella cartella "Analisi_DBs" viene fatta un'analisi di ogni database utilizzato.

Il file "MODELLO_NOTEBOOK" è stato utilizzato come modello per creare
cicli di modelli per la ricerca di parametri migliori. Implementa un modello
bidirezionale CuDNLSTM su immagini 256x64 interpolate con accuratezza finale ~93%.

Il file "MODELLO_LETTURA_FACILE" è un modello per la lettura facile dei risultati,
nascondendo l'implementazione.

Il file "Bidirectional_methods_test" contiene i risultati di 
4 modelli bidirezionali implementati in 4 diversi modi:
concat, sum, mul e ave. Testato su immagini 64x64 croppate.


Il file "Interpolation_methods_Test" contiene i risultati dei vari
metodi di interpolazione che cv2 offre:
INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4.

Nella cartella "Search_best_models" ci sono i file "Searching_1-6" e "Searching_7-8";
 contengono il lancio di diversi modelli al fine di testare diversi parametri;
 contiene anche i risultati del search.