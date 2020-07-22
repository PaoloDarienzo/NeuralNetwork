Il file Vanilla_18k_64x64_truncated implementa un
modello di rete neurale shallow, ovvero 
2 layer fully connected da 141 e 94 nodi.
Le immagini sono 0-padded o croppate in 64x64.
L'accuratezza finale è attorno all'81%.
(Patience 10)
Top 5 set offuscamenti
('EncodeLiterals-InitOpaque-Split', 34)
('Flatten-RandomFuns-Split', 31)
('InitEntropy-InitOpaque-RandomFuns', 29)
('Flatten-InitOpaque-RandomFuns', 25)
('EncodeArithmetic-Flatten-RandomFuns', 23)
Interessante notare che la singola trasformazione
InitEntropy-InitOpaque-RandomFuns svetta rispetto alle
altre trasformazioni, come singola trasformazione/errore.



Il file Vanilla_18k_64x64 implementa un
modello di rete neurale shallow, ovvero 
2 layer fully connected da 141 e 94 nodi.
Le immagini sono interpolate nella dimensione 64x64.
L'accuratezza finale è attorno al 90%.
Top 5 set offuscamenti
('Flatten-RandomFuns-Split', 27)
('EncodeArithmetic-Flatten-RandomFuns', 21)
('EncodeLiterals-InitOpaque-Split', 16)
('EncodeLiterals-Flatten-InitOpaque', 16)
('EncodeLiterals-RandomFuns-Split', 14)


Il file Vanilla_18k_596x64_truncated  implementa un
modello di rete neurale shallow, ovvero 
2 layer fully connected da 141 e 94 nodi.
Le immagini sono 0-padded o croppate in 596x64.
L'accuratezza finale è attorno al 90%.
Top 5 set offuscamenti
('Flatten-RandomFuns-Split', 33)
('EncodeLiterals-InitOpaque-Split', 21)
('EncodeLiterals-InitOpaque-RandomFuns', 18)
('EncodeArithmetic-RandomFuns-Split', 18)
('EncodeArithmetic-Flatten-RandomFuns', 17)
