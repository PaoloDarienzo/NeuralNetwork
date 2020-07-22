Il file "Bi_CudNN_18k_64x596" contiene una rete bi-direzionale LSTM, con immagini 
troncate (o 0-padded) della dimensione 64x596.
Su database 18k.
1.   2 layer da 141 nodi, 94 nodi;
1.   2 layer da 141 nodi, 94 nodi;
2.   Immagini 64x596, troncate;
3.   512 batch size;
4.   Accuratezza finale: 92,89%.
5.   Tempo: 250.
6.   Epochs: 54.
(Patience 10)
Lista offuscamenti che causano più errori:
('Flatten-RandomFuns-Split', 33)
('EncodeArithmetic-Flatten-RandomFuns', 19)
('EncodeLiterals-RandomFuns-Split', 16)
('Flatten-InitOpaque-RandomFuns', 11)
('EncodeArithmetic-Flatten-Split', 11)
====

Il file "CudNN_18k_64x596" contiene una rete NON bi-direzionale LSTM, con immagini 
troncate (o 0-padded) della dimensione 64x596.
Su database 18k.
1.   2 layer da 141 nodi, 94 nodi;
2.   Immagini 64x596, troncate;
3.   256 batch size;
4.   Accuratezza finale: 72,81.
5.   Tempo: 519.
6.   Epochs: 100.
(Patience 5)
Lista offuscamenti che causano più errori:
('Flatten-RandomFuns-Split', 33)
('EncodeLiterals-InitOpaque-RandomFuns', 30)
('Flatten-InitOpaque-RandomFuns', 28)
('EncodeLiterals-Flatten-Split', 28)
('EncodeArithmetic-EncodeLiterals-InitOpaque', 28)

==========================================

Il file "Bi_CudNN_18k_596x64" contiene una rete bi-direzionale LSTM, con immagini 
troncate (o 0-padded) della dimensione 596x64.
Su database 18k.
2.   Immagini 596x64, troncate;
3.   512 batch size;
4.   Accuratezza finale: 82,78%.
5.   Tempo: 1309.
6.   Epochs: 100.
(Patience 10)
Lista offuscamenti che causano più errori:
('EncodeLiterals-InitOpaque-RandomFuns', 22)
('InitEntropy-InitOpaque-RandomFuns', 22)
('Flatten-InitOpaque-RandomFuns', 21)
('Flatten-RandomFuns-Split', 20)
('EncodeLiterals-RandomFuns-Split', 19)

====

Il file "CudNN_18k_596x64" contiene una rete NON bi-direzionale LSTM, con immagini 
troncate (o 0-padded) della dimensione 596x64.
Su database 18k.
1.   2 layer da 141 nodi, 94 nodi;
2.   Immagini 596x64, troncate;
3.   256 batch size;
(Patience 10)
Non impara nulla.