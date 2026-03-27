# EFFICIENTNET_ARTEFACTOS

Modulo de `transfer learning` para clasificar artefactos arqueologicos a partir de pares `(foto, dibujo)` del mismo `itemUUID`.

## Idea

- Parte de `EfficientNet-B0` preentrenada en ImageNet.
- Congela primero el backbone y entrena solo la cabeza de clasificacion.
- Despues desbloquea los ultimos bloques para un ajuste fino.
- Fusiona las dos modalidades con un backbone compartido:
  - embedding de la foto
  - embedding del dibujo
  - diferencia absoluta entre embeddings
  - producto elemento a elemento
- Mantiene la misma finalidad practica del `CVAE`: clasificar el artefacto usando el par `foto+dibujo`.

## Salidas

- `checkpoints/`: pesos entrenados y prototipos del embedding
- `historial/`: metricas por epoca en JSON
- `pruebas visuales/`: ejemplos visuales del split de test

## Entrenamiento

Entrenamiento normal con pesos preentrenados:

```bash
python PROPIOS/EFFICIENTNET_ARTEFACTOS/efficientnet_artefactos.py train --split-name 80-20 --epochs 10 --freeze-epochs 3 --unfreeze-blocks 2
```

Prueba rapida local sin descargar pesos:

```bash
python PROPIOS/EFFICIENTNET_ARTEFACTOS/efficientnet_artefactos.py train --split-name 80-20 --epochs 1 --weights none --image-size 128 --batch-size 2 --max-train-samples 8 --max-test-samples 4 --visual-examples 2 --force-cpu
```

## Prediccion

```bash
python PROPIOS/EFFICIENTNET_ARTEFACTOS/efficientnet_artefactos.py predict "ruta\\a\\photo.tif" "ruta\\a\\drawing.tif" "PROPIOS\\EFFICIENTNET_ARTEFACTOS\\checkpoints\\efficientnet_pair_80-20_experiment_0.pt"
```

## Pruebas visuales

```bash
python PROPIOS/EFFICIENTNET_ARTEFACTOS/efficientnet_artefactos.py visual-tests "PROPIOS\\EFFICIENTNET_ARTEFACTOS\\checkpoints\\efficientnet_pair_80-20_experiment_0.pt" --num-examples 8
```

## Nota sobre los pesos preentrenados

Por defecto usa `EfficientNet-B0` con pesos de ImageNet. Si el entorno no tiene esos pesos en cache y no puede descargarlos, usa `--weights none` para una prueba tecnica, aunque eso ya no seria transfer learning completo.
