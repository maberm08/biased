# CNN_ARTEFACTOS

Modulo ligero para clasificacion de artefactos arqueologicos sobre `cssl_dataset`.

## Idea

- Usa el formato de `split.txt` ya presente en el repositorio.
- Trabaja sobre `cssl_dataset/all_image_base/1` o `cssl_dataset/all_drawing_base/1`.
- Entrena tres modelos pequenos: `CNN`, `MLP` y `KAN`.
- Usa por defecto entradas de `256x256`.
- Guarda checkpoints reutilizables e historiales en JSON.
- Puede crear un GIF comparando las curvas de loss.

## Ficheros generados

- `cnn_artefactos.py`: entrenamiento y prediccion.
- `checkpoints/`: pesos guardados.
- `historial/`: metricas por epoca en JSON.
- `gifs/`: comparativas animadas de loss.

## Entrenar un modelo

```bash
python PROPIOS/CNN_ARTEFACTOS/cnn_artefactos.py train --model-type cnn --split-name 50-50-q --modality photos --epochs 10 --max-train-samples 400 --max-test-samples 200
python PROPIOS/CNN_ARTEFACTOS/cnn_artefactos.py train --model-type mlp --split-name 50-50-q --modality photos --epochs 10 --max-train-samples 400 --max-test-samples 200
python PROPIOS/CNN_ARTEFACTOS/cnn_artefactos.py train --model-type kan --split-name 50-50-q --modality photos --epochs 10 --max-train-samples 400 --max-test-samples 200
```

## Entrenar las tres y generar el GIF

```bash
python PROPIOS/CNN_ARTEFACTOS/cnn_artefactos.py train-all --split-name 50-50-q --modality photos --epochs 10 --max-train-samples 400 --max-test-samples 200
```

## Regenerar GIFs desde historiales

```bash
python PROPIOS/CNN_ARTEFACTOS/cnn_artefactos.py gifs-from-history PROPIOS/CNN_ARTEFACTOS/historial/cnn_drawings_50-50_experiment_0.json PROPIOS/CNN_ARTEFACTOS/historial/mlp_drawings_50-50_experiment_0.json PROPIOS/CNN_ARTEFACTOS/historial/kan_drawings_50-50_experiment_0.json
```

## Ejemplo de prediccion

```bash
python PROPIOS/CNN_ARTEFACTOS/cnn_artefactos.py predict "ruta/a/imagen.tif" "PROPIOS/CNN_ARTEFACTOS/checkpoints/cnn_photos_50-50-q_experiment_0.pt"
```

## Nota

La configuracion por defecto usa `50-50-q` y un subconjunto del split para abaratar el entrenamiento. Si quieres mas datos, sube `--max-train-samples`, `--max-test-samples` o cambia el split a `50-50`.
